# ==============================================================================
# Database Initialization: Schema Definition ONLY (Zero Hardcoded Records)
# ==============================================================================
def init_db():
    with db_conn() as conn:
        # 1. Audit Transaction Log
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

        # 2. Schema: Dishes (Drop if incompatible to resolve column mismatch crashes)
        dish_cols = [c[1] for c in conn.execute("PRAGMA table_info(master_dishes_db)").fetchall()]
        if dish_cols and len(dish_cols) < 4:
            conn.execute("DROP TABLE IF EXISTS master_dishes_db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_dishes_db (
                dish_id TEXT PRIMARY KEY,
                name TEXT,
                main_carb TEXT,
                protein TEXT
            )
        """)

        # 3. Schema: Branches
        branch_cols = [c[1] for c in conn.execute("PRAGMA table_info(master_branches_db)").fetchall()]
        if branch_cols and len(branch_cols) < 6:
            conn.execute("DROP TABLE IF EXISTS master_branches_db")
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

        # 4. Schema: Rewards
        reward_cols = [c[1] for c in conn.execute("PRAGMA table_info(master_rewards_db)").fetchall()]
        if reward_cols and len(reward_cols) < 6:
            conn.execute("DROP TABLE IF EXISTS master_rewards_db")
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
        
        # Save Google Sheets Endpoints (No food or branch data inserted here)
        base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSloK2WPNFd8HPY4RfL2rNhwhk_kD12H0q09nDcrlMrx5O_zqslCOi1TPAXvlHtnP1FWxyJxGgG99QX/pub?output=csv"
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('base_url', ?)", (base_url,))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('dishes_url', ?)", (f"{base_url}&sheet=dishes",))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('branches_url', ?)", (f"{base_url}&sheet=branches",))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('rewards_url', ?)", (f"{base_url}&sheet=rewards",))
