SECTION_WIDTH = 70


def log_info(message: str):
    print(f"ℹ️ [INFO] {message}")


def log_step(message: str):
    print(f"🔹 [STEP] {message}")


def log_ok(message: str):
    print(f"✅ [OK] {message}")


def log_warn(message: str):
    print(f"⚠️ [WARN] {message}")


def log_error(message: str):
    print(f"❌ [ERROR] {message}")


def log_section(title: str, width: int = SECTION_WIDTH):
    line = "=" * width
    print(f"\n{line}")
    print(f"📌 [SECTION] {title}")
    print(line)
