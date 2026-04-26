@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    print("FILE SIZE:", len(content))

    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("IMG:", img is not None)

    if img is None:
        return {"error": "IMAGE_DECODE_FAILED"}

    clean = preprocess(img)
    text = extract_text(clean)

    ticker_img = crop_ticker_area(img)
    ticker_text = extract_text(preprocess(ticker_img))
    ticker = detect_ticker(ticker_text)

    if ticker == "UNKNOWN":
        ticker = detect_ticker(text)

    interval = detect_interval(text)

    O = find_number(text, "O")
    H = find_number(text, "H")
    L = find_number(text, "L")
    C = find_number(text, "C")

    MA20 = multi_find(text, ["MA20", "MA 20", "SMA20"])
    EMA9 = multi_find(text, ["EMA9", "EMA 9"])
    SMA50 = multi_find(text, ["SMA50", "SMA 50"])
    DEMA9 = multi_find(text, ["DEMA9", "DEMA 9"])
    RSI = multi_find(text, ["RSI"])
    VOL = multi_find(text, ["WOLUMEN", "VOLUME", "VOL", "WOL"])
    RVOL = multi_find(text, ["RVOL", "R-VOL", "REL VOL"])

    return {
        "ticker": ticker,
        "interval": interval,
        "O": O,
        "H": H,
        "L": L,
        "C": C,
        "MA20": MA20,
        "EMA9": EMA9,
        "DEMA9": DEMA9,
        "SMA50": SMA50,
        "RSI": RSI,
        "VOL": VOL,
        "RVOL": RVOL,
    }
