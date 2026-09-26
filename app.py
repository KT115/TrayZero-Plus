import io
import urllib.request

# ==============================================================================
# 2. Database Initialization: Schema Definition ONLY (Zero Hardcoded Records)
# ==============================================================================
def init_db():
    with db_conn() as conn:
        # 1. 審計流水日誌表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                audit_date TEXT,
                audit_month TEXT,
                branch_name TEXT,
                member_id TEXT,
                dish_name TEXT,
                primary_waste TEXT,
                waste_ratio REAL,
                cost_waste_hkd REAL,
                co2_emission_kg REAL,
                reward_issued TEXT
            )
        """)

        # 2. Schema: Dishes (純結構，不寫死任何菜名)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_dishes_db (
                dish_id TEXT PRIMARY KEY,
                name TEXT,
                main_carb TEXT,
                protein TEXT
            )
        """)

        # 3. Schema: Branches (純結構，不寫死任何分店)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_branches_db (
                name TEXT PRIMARY KEY,
                level TEXT,
                district TEXT,
                traffic TEXT,
                avg_covers INTEGER,
                base_rice_g INTEGER
            )
        """)

        # 4. Schema: Rewards (純結構，不寫死任何獎勵階梯)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_rewards_db (
                reward_id TEXT PRIMARY KEY,
                tier_name TEXT,
                max_waste_ratio REAL,
                reward_type TEXT,
                reward_description TEXT,
                is_active INTEGER
            )
        """)

        # 5. Cloud Endpoint Configuration Storage
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cloud_config_db (
                key TEXT PRIMARY KEY,
                url TEXT
            )
        """)

        # 記錄 Google Sheet CSV 發布基本網址
        base_gdrive_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSloK2WPNFd8HPY4RfL2rNhwhk_kD12H0q09nDcrlMrx5O_zqslCOi1TPAXvlHtnP1FWxyJxGgG99QX/pub?output=csv"
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('base_url', ?)", (base_gdrive_url,))


def fetch_csv_safely(url):
    """防封鎖、防連線逾時的 Google 雲端 CSV 抓取輔助函數"""
    if not url or not str(url).startswith("http"):
        return None
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            content = response.read().decode('utf-8-sig')
            df = pd.read_csv(io.StringIO(content))
            return df
    except Exception as e:
        print(f"Fetch cloud CSV error from {url}: {e}")
        return None


# ==============================================================================
# 3. Pure Cloud-Driven Data Loaders (Zero Hardcoded Data)
# ==============================================================================
def get_live_dishes():
    cloud_urls = get_cloud_urls()
    url = cloud_urls.get("dishes_url") or cloud_urls.get("base_url")

    # 1. 優先從 Google Sheets 抓取
    cloud_df = fetch_csv_safely(url)
    if cloud_df is not None and not cloud_df.empty and "name" in cloud_df.columns:
        cloud_df = cloud_df.dropna(subset=["name"])
        cloud_df = cloud_df[cloud_df["name"].astype(str).str.strip() != ""]
        cloud_df.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            cloud_df.to_sql("master_dishes_db", conn, if_exists="replace", index=False)
        return cloud_df

    # 2. 容錯讀取：本地檔案快取
    if os.path.exists(DISH_FILE):
        try:
            df = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
            if not df.empty and "name" in df.columns:
                return df
        except Exception:
            pass

    # 3. 容錯讀取：SQLite 庫存
    with db_conn() as conn:
        db_df = pd.read_sql("SELECT * FROM master_dishes_db", conn)
        if not db_df.empty and "name" in db_df.columns:
            return db_df

    # 4. 若全空，回傳具有完整欄位的空 DataFrame（防崩潰）
    return pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])


def get_live_branches():
    cloud_urls = get_cloud_urls()
    url = cloud_urls.get("branches_url") or cloud_urls.get("base_url")

    cloud_df = fetch_csv_safely(url)
    if cloud_df is not None and not cloud_df.empty and "name" in cloud_df.columns:
        cloud_df = cloud_df.dropna(subset=["name"])
        cloud_df = cloud_df[cloud_df["name"].astype(str).str.strip() != ""]
        cloud_df.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            cloud_df.to_sql("master_branches_db", conn, if_exists="replace", index=False)
        return cloud_df

    if os.path.exists(BRANCH_FILE):
        try:
            df = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
            if not df.empty and "name" in df.columns:
                return df
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT * FROM master_branches_db", conn)
        if not db_df.empty and "name" in db_df.columns:
            return db_df

    return pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g"])


def get_live_rewards():
    cloud_urls = get_cloud_urls()
    url = cloud_urls.get("rewards_url") or cloud_urls.get("base_url")

    cloud_df = fetch_csv_safely(url)
    if cloud_df is not None and not cloud_df.empty and "tier_name" in cloud_df.columns:
        cloud_df = cloud_df.dropna(subset=["tier_name"])
        cloud_df = cloud_df[cloud_df["tier_name"].astype(str).str.strip() != ""]
        cloud_df.to_csv(REWARD_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            cloud_df.to_sql("master_rewards_db", conn, if_exists="replace", index=False)
        return cloud_df

    if os.path.exists(REWARD_FILE):
        try:
            df = pd.read_csv(REWARD_FILE, encoding="utf-8-sig")
            if not df.empty and "tier_name" in df.columns:
                return df
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT * FROM master_rewards_db", conn)
        if not db_df.empty and "tier_name" in db_df.columns:
            return db_df

    return pd.DataFrame(columns=["reward_id", "tier_name", "max_waste_ratio", "reward_type", "reward_description", "is_active"])
