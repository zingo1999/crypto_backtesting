import datetime
import sys

import pandas as pd
from py_vollib.black_scholes.greeks.analytical import delta
import time
import requests

# 1. 拉 BTC options 列表
instruments = requests.get(
    "https://www.deribit.com/api/v2/public/get_instruments",
    params={"currency": "BTC", "kind": "option", "expired": "false"}
).json()["result"]

# 2. 選擇一個到期日（例如最近一個月）
expiry_list = sorted(list(set(i["expiration_timestamp"] for i in instruments)))
# target_expiry = expiry[0]

now = int(time.time())
six_months_later = now + 6 * 30 * 24 * 60 * 60  # 大約6個月（180日）
expiry_list_s = [e // 1000 for e in expiry_list]
target_expiry = min(expiry_list_s, key=lambda x: abs(x - six_months_later))
target_expiry = target_expiry * 1000

options = [i for i in instruments if i["expiration_timestamp"] == target_expiry]

# 3. 拉每個 option 嘅數據
def get_summary(instr):
    response = requests.get(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument",
        params={"instrument_name": instr["instrument_name"]}
    ).json()["result"]
    if not response:
        return None
    r = response[0]
    return {
        "name": instr["instrument_name"],
        "strike": instr["strike"],
        "option_type": instr["option_type"],
        "iv": r["mark_iv"],
        "underlying_price": r["underlying_price"],
        "interest_rate": r["interest_rate"],
        "expiration_timestamp": instr["expiration_timestamp"]
    }


data = []
for i, opt in enumerate(options):
    if opt["option_type"] in ["call", "put"]:
        result = get_summary(opt)
        if result is not None:
            data.append(result)
    if (i + 1) % 5 == 0:
        start_time = time.time()

        # 即時計算delta同skew
        for d in data:
            S = d["underlying_price"]
            K = d["strike"]
            t = (d["expiration_timestamp"] / 1000 - time.time()) / (365 * 24 * 60 * 60)
            rfr = d["interest_rate"]
            sigma = d["iv"] / 100
            flag = 'c' if d["option_type"] == "call" else 'p'
            d["delta"] = delta(flag, S, K, t, rfr, sigma)

        calls = [d for d in data if d["option_type"] == "call"]
        puts = [d for d in data if d["option_type"] == "put"]

        call_delta_values = [call['delta'] for call in calls]
        put_delta_values = [put['delta'] for put in puts]

        call_candidates = [c for c in calls if c["delta"] <= 0.25]
        put_candidates = [p for p in puts if p["delta"] >= -0.25]

        if call_candidates and put_candidates:
            call_25 = min(call_candidates, key=lambda x: abs(x["delta"] - 0.25))
            put_25 = min(put_candidates, key=lambda x: abs(x["delta"] + 0.25))
            skew = put_25["iv"] - call_25["iv"]
            print(f"25 Delta Skew: {skew:.2f}")
            break
        else:
            print("冇搵到合適delta，直接sleep")

        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed < 10:
            print(f"Sleep {10 - elapsed:.2f} 秒")
            time.sleep(10 - elapsed)

########
# 主程式 loop options，控制 rate limit
# data = []
# for i, opt in enumerate(options):
#     if opt["option_type"] in ["call", "put"]:
#         result = get_summary(opt)
#         if result is not None:
#             data.append(result)
#     if (i + 1) % 5 == 0:
#         time.sleep(10)
#
# for d in data:
#     S = d["underlying_price"]
#     K = d["strike"]
#     t = (d["expiration_timestamp"]/1000 - time.time()) / (365*24*60*60)
#     rfr = d["interest_rate"]
#     sigma = d["iv"] / 100  # 轉做小數
#     flag = 'c' if d["option_type"] == "call" else 'p'
#     d["delta"] = delta(flag, S, K, t, rfr, sigma)
#
# calls = [d for d in data if d["option_type"] == "call"]
# puts = [d for d in data if d["option_type"] == "put"]
#
# call_25 = min(calls, key=lambda x: abs(x["delta"] - 0.25))
# put_25 = min(puts, key=lambda x: abs(x["delta"] + 0.25))
#
# skew = put_25["iv"] - call_25["iv"]
# print(f"25 Delta Skew: {skew:.2f}")




from config import Credentials
import pandas as pd
data_source_key = Credentials('glassnode').api_key
params = {
    'a': 'BTC',
    'e': 'deribit',
    'i': '1h',
    'api_key': data_source_key,
}
res = requests.get('https://api.glassnode.com/v1/metrics/derivatives/options_25delta_skew_6_months', params=params)
data = res.json()
df = pd.DataFrame(data)

sys.exit()




class Student:
    def __init__(self, name, score):
        self.name = name
        self._score = score
        self.change_count = 0
        self.level = self._calc_level(score)

    @property
    def score(self):
        return self._score + 5

    @score.setter
    def score(self, value):
        if value < 0 or value > 100:
            raise ValueError("分數要介乎 0 至 100")
        self._score = value
        self.change_count += 1
        self.level = self._calc_level(value)
        print(f"{self.name} 嘅分數已更新為 {value}（已更改 {self.change_count} 次），等級：{self.level}")

    @property
    def pass_status(self):
        return "合格" if self.score >= 60 else "不合格"

    def _calc_level(self, score):
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        else:
            return "C"


s1 = Student('小明', 77)
s2 = Student('李家誠', 50)
pass


class Testing:
    def __init__(self, data_source):
        self._api_key = {
            'glassnode': 'd8912jd',
            'coinglass': 'iehno128asd'
        }
        if data_source in self._api_key:
            self.selected_api_key = self._api_key[data_source]
        else:
            raise ValueError("Invalid data source provided.")

    @property
    def api_key(self):
        """Getter for the selected API key."""
        return self.selected_api_key

    @api_key.setter
    def api_key(self, value):
        """Setter for the selected API key."""
        raise AttributeError("Cannot modify the API key directly.")

# Example usage
data_source = 'glassnode'
testing_instance = Testing(data_source)

# Accessing the API key
print(testing_instance.api_key)  # Output: d8912jd

# Attempting to set the API key directly will raise an error
try:
    testing_instance.api_key = 'new_key'
except AttributeError as e:
    print(e)  # Output: Cannot modify the API key directly.