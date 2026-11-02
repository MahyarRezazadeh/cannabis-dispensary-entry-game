"""
Export the best-available-frequency revenue panel for each of the 5 sample
states to .dta, for the New Mexico project folder. Per explicit user
request: New Mexico's true dispensary-level MONTHLY panel, plus each other
state's data at whatever the finest frequency actually available is (not
forced to match NM's granularity, since it doesn't exist for them):
  - Colorado: county-MONTH (CO Dept. of Revenue raw file, 2014-2026),
    including retailer counts already present in that source.
  - Washington: county-FISCAL-YEAR (WSLCB workbooks report whole-FY
    "Reporting Period" strings, e.g. "07/01/2017 - 06/30/2018" -- no finer
    frequency exists in what this project has).
  - Oregon: county-MONTH, rebuilt from the raw OLCC transaction-level pull
    (oregon_olcc_raw.csv, monthly dates 2016-10 through 2026-08) -- finer
    than the county-YEAR file used in the actual analysis panel.
  - Montana: county-MONTH (MT DOR scraped pages, 2022-2026).
  - New Mexico: dispensary-MONTH (already built, nm_dispensary_panel_monthly.csv).

Stata .dta export requires ASCII-safe column names (no special characters)
and no NaN in string columns, handled below.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project"


def save_dta(df, name):
    path = os.path.join(OUT, name)
    df.to_stata(path, write_index=False, version=118)
    print(f"Wrote {path}: {df.shape}")


def nm():
    df = pd.read_csv(os.path.join(BASE, "tables", "nm_dispensary_panel_monthly.csv"))
    save_dta(df, "nm_dispensary_panel_monthly.dta")


def co():
    raw = pd.read_csv(os.path.join(BASE, "border revenue", "data", "raw", "co_marijuana_sales_by_county.csv"))
    raw = raw.dropna(subset=["year", "month", "county"])
    raw["year"] = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)
    raw["med_sales"] = pd.to_numeric(raw["med_sales"], errors="coerce").fillna(0)
    raw["rec_sales"] = pd.to_numeric(raw["rec_sales"], errors="coerce").fillna(0)
    raw["total_sales"] = raw["med_sales"] + raw["rec_sales"]
    out = raw[["county", "year", "month", "med_sales", "rec_sales", "total_sales",
               "numberofmedicalmarijuanaretailers", "numberofretailmarijuanaretailers"]].copy()
    out = out.rename(columns={"numberofmedicalmarijuanaretailers": "n_medical_retailers",
                               "numberofretailmarijuanaretailers": "n_retail_retailers"})
    out["state_abbr"] = "CO"
    save_dta(out, "co_revenue_panel_monthly.dta")


def wa():
    import openpyxl

    def read_sheet(path, sheet, fy_year):
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb[sheet]
        header_row = None
        rows = []
        for row in ws.iter_rows(values_only=True):
            if row[0] == "County":
                header_row = True
                continue
            if not header_row:
                continue
            county, period, total_sales, excise_tax = row[0], row[1], row[2], row[3]
            if county is None or total_sales is None:
                continue
            rows.append({"county": str(county).strip().title(), "fiscal_year": fy_year,
                         "reporting_period": period, "total_sales": total_sales, "excise_tax": excise_tax})
        return rows

    WA_DIR = os.path.join(BASE, "border revenue", "data", "raw")
    rows = []
    rows += read_sheet(os.path.join(WA_DIR, "wa_fy15.xlsx"), "Retailer", 2015)
    rows += read_sheet(os.path.join(WA_DIR, "wa_fy16.xlsx"), "Retailer", 2016)
    rows += read_sheet(os.path.join(WA_DIR, "wa_fy17.xlsx"), "Retailer", 2017)
    for fy in range(2018, 2025):
        rows += read_sheet(os.path.join(WA_DIR, "wa_fy2018_2024.xlsx"), f"FY{fy % 100}", fy)
    rows += read_sheet(os.path.join(WA_DIR, "wa_fy25.xlsx"), "FY25", 2025)

    out = pd.DataFrame(rows)
    out["state_abbr"] = "WA"
    save_dta(out, "wa_revenue_panel_fiscalyear.dta")


def orr():
    raw = pd.read_csv(os.path.join(BASE, "data", "raw", "oregon_revenue", "oregon_olcc_raw.csv"))
    raw["date"] = pd.to_datetime(raw["date"])
    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    raw["sales"] = pd.to_numeric(raw["sales"], errors="coerce").fillna(0)
    monthly = raw.groupby(["county", "year", "month"], as_index=False)["sales"].sum()
    monthly = monthly.rename(columns={"sales": "total_sales"})
    monthly["state_abbr"] = "OR"
    save_dta(monthly, "or_revenue_panel_monthly.dta")


def mt():
    raw = pd.read_csv(os.path.join(BASE, "data", "raw", "montana_revenue", "montana_county_month_sales_clean.csv"))
    raw = raw.rename(columns={"sales": "total_sales"})
    raw["state_abbr"] = "MT"
    save_dta(raw, "mt_revenue_panel_monthly.dta")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    nm()
    co()
    wa()
    orr()
    mt()
