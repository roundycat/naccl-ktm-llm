import sys, importlib.util, json
spec = importlib.util.spec_from_file_location("g3", "tune_gate3.py")
# tune_gate3 는 실행형이라 재사용 대신 필요한 로직만 복제
