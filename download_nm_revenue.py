"""
Download New Mexico CROP (Cannabis Reporting Online Portal) monthly sales files.
Source: https://crop.rld.nm.gov/data-catalog.html
Discovered API (via browser network inspection, no auth required):
  GetWidgetFiles     -> lists files (fileId, title) for a given widgetId/folderId
  GetWidgetFileLink  -> resolves a fileId to an actual signed download URL
"""
import json
import time
import urllib.request
import urllib.parse
import csv
import os

API_BASE = "https://klvg4oyd4j.execute-api.us-west-2.amazonaws.com/prod"
ACCOUNT_GUID = "1ee897135beb4b1c82715d36398de4c5"
OUT_DIR = os.path.join("data", "raw", "newmexico_revenue")

# (year, widgetId, rootFolderId, folderId) discovered by clicking each year tab
YEAR_FOLDERS = [
    (2022, "8c66a566-31bc-4e87-89cf-b098dfb6a570", "fa7004d3-f466-482d-97b2-7b57648d8b6d", "fa7004d3-f466-482d-97b2-7b57648d8b6d"),
    (2023, "8f303ea3-65f3-4edf-b50a-b60964529c0b", "f0c2ac93-4afd-479a-8410-16b5924562f7", "f0c2ac93-4afd-479a-8410-16b5924562f7"),
    (2024, "b9dd339b-de1d-406b-abaf-c30c1f3d783a", "9957fddd-766f-4333-af61-cc5e13e85951", "1ae70aa3-5f89-41d7-bb2c-ca2b194019b0"),
    (2025, "b9dd339b-de1d-406b-abaf-c30c1f3d783a", "9957fddd-766f-4333-af61-cc5e13e85951", "dd39327e-357c-4576-8ee6-80cbdb42f5de"),
    (2026, "b9dd339b-de1d-406b-abaf-c30c1f3d783a", "9957fddd-766f-4333-af61-cc5e13e85951", "e1433770-23dd-42d9-a249-f0063ccb99bb"),
]


def api_get(path, params):
    url = f"{API_BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_files(widget_id, root_folder_id, folder_id):
    data = api_get("GetWidgetFiles", {
        "widgetId": widget_id,
        "rootFolderId": root_folder_id,
        "folderId": folder_id,
        "authTokenGUID": "",
        "accountGUID": ACCOUNT_GUID,
    })
    return data["data"]["files"]


def get_file_link(widget_id, file_id):
    data = api_get("GetWidgetFileLink", {
        "fileId": file_id,
        "widgetId": widget_id,
        "authTokenGUID": "",
        "accountGUID": ACCOUNT_GUID,
    })
    # response shape observed: {"status":"ok","data":{"url": "..."}} -- fall back to raw dump if different
    d = data.get("data")
    if isinstance(d, dict):
        for key in ("url", "link", "fileUrl", "downloadUrl"):
            if key in d:
                return d[key]
    if isinstance(d, str):
        return d
    raise RuntimeError(f"Unrecognized GetWidgetFileLink response shape: {data}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []

    for year, widget_id, root_folder_id, folder_id in YEAR_FOLDERS:
        print(f"--- {year} ---")
        files = list_files(widget_id, root_folder_id, folder_id)
        print(f"  {len(files)} files listed")
        for f in files:
            title = f.get("title") or os.path.splitext(f.get("name") or f["fileId"])[0]
            file_id = f["fileId"]
            orig_name = f["name"]
            ext = os.path.splitext(orig_name)[1] or ".xls"
            safe_title = title.replace(" ", "_").replace("-", "").replace("__", "_")
            out_name = f"{year}_{safe_title}{ext}"
            out_path = os.path.join(OUT_DIR, out_name)

            if os.path.exists(out_path):
                print(f"  [skip, exists] {out_name}")
            else:
                try:
                    link = get_file_link(widget_id, file_id)
                    urllib.request.urlretrieve(link, out_path)
                    print(f"  [ok] {out_name}")
                    time.sleep(0.3)
                except Exception as e:
                    print(f"  [FAIL] {title}: {e}")
                    out_name = ""

            manifest_rows.append({
                "year": year, "title": title, "orig_name": orig_name,
                "fileId": file_id, "local_file": out_name,
            })

    manifest_path = os.path.join(OUT_DIR, "_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["year", "title", "orig_name", "fileId", "local_file"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest written: {manifest_path} ({len(manifest_rows)} files)")


if __name__ == "__main__":
    main()
