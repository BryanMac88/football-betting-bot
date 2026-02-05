def set_visible_tabs(keep_titles: list[str]) -> None:
    """
    Hide all worksheets except keep_titles.
    Uses Google Sheets batchUpdate so it works even if gspread hide/show helpers differ by version.
    """
    sh = _open_sheet(env("SHEET_ID"))  # uses your existing _open_sheet()

    meta = sh.fetch_sheet_metadata()
    sheets = meta.get("sheets", [])

    requests = []
    for s in sheets:
        props = s.get("properties", {})
        title = props.get("title")
        sheet_id = props.get("sheetId")
        if title is None or sheet_id is None:
            continue

        hidden = title not in keep_titles
        requests.append({
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "hidden": hidden
                },
                "fields": "hidden"
            }
        })

    if requests:
        sh.batch_update({"requests": requests})
