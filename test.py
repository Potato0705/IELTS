import os, requests, datetime, sys, json

API_KEY = os.getenv("OPENROUTER_API_KEY")
assert API_KEY, "No OPENROUTER_API_KEY env!"

BASE = "https://openrouter.ai"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://ielts-alphaevolve.local",
    "X-Title": "IELTS-AlphaEvolve",
}

def get_free_used_today():
    # 1) Activity 不带 date 参数
    url = BASE + "/api/v1/activity"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])

    today_utc = datetime.datetime.now(datetime.UTC).date()

    free_used = 0

    # 2) 兼容两种常见结构：
    #   A. 按天汇总: item 有 date/day 字段
    #   B. 按请求明细: item 有 timestamp 字段
    for item in data:
        model = str(item.get("model", ""))

        # A) daily buckets
        if "date" in item or "day" in item:
            d_str = item.get("date") or item.get("day")
            try:
                d = datetime.date.fromisoformat(d_str)
            except Exception:
                continue
            if d == today_utc and model.endswith(":free"):
                free_used += int(item.get("requests", 0))
            continue

        # B) per-request logs
        if "timestamp" in item:
            try:
                ts = item["timestamp"]
                # timestamp 可能是秒或毫秒
                if ts > 1e12:
                    ts /= 1000.0
                d = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).date()
            except Exception:
                continue
            if d == today_utc and model.endswith(":free"):
                free_used += 1

    return int(free_used), str(today_utc)

def get_key_limits_and_credits():
    # 这个接口可以看“每分钟剩余请求数 / credits 余额”等实时信息
    # 但不会直接给你“free 今日剩余次数”
    url = BASE + "/api/v1/key"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    try:
        free_used, today = get_free_used_today()
        # 你余额≥$10 => free daily cap = 1000
        daily_cap = 1000
        print("UTC today:", today)
        print("free requests used today:", free_used)
        print("free remaining today:", max(daily_cap - free_used, 0))
    except Exception as e:
        print("Failed to read activity:", repr(e))
        print("Raw activity for debugging:")
        try:
            raw = requests.get(BASE + "/api/v1/activity", headers=headers, timeout=30).json()
            print(json.dumps(raw, indent=2)[:2000])
        except Exception as e2:
            print("Also failed to dump raw:", repr(e2))

    try:
        key_info = get_key_limits_and_credits()
        print("\n/api/v1/key info (rate limits & credits):")
        print(json.dumps(key_info, indent=2)[:2000])
    except Exception as e:
        print("Failed to read /key:", repr(e))
