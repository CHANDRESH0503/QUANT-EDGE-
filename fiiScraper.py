import os
import time
import traceback
import pandas as pd
from playwright.sync_api import sync_playwright

OUTPUT_DIR = "dii_cash_daily_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_and_save_data(data_list, year):
    """Saves the completed year's data instantly to CSV."""
    print(f"\n[SYSTEM] Saving {len(data_list)} rows for the year {year}...")
    if not data_list:
        print(f"⚠️ No data collected for {year}.")
        return

    columns = [
        "Date", "MF_Equity_Gross_Purchase", "MF_Equity_Gross_Sales",
        "MF_Equity_Net_Purchase_Sales", "MF_Debt_Net_Purchase_Sales",
        "MF_Debt_Gross_Sales", "MF_Debt_Gross_Purchase",
    ]

    df = pd.DataFrame(data_list, columns=columns)

    try:
        df["_sort_date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
        df = df.sort_values(by="_sort_date").drop(columns=["_sort_date"])
    except Exception:
        pass

    file_path = os.path.join(OUTPUT_DIR, f"dii_Cash_Daily_{year}.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Successfully saved {year} to {file_path}\n")


def scrape_mf_cash():
    target_url = "https://trendlyne.com/macro-data/fii-dii/month/fii-month/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = context.new_page()

        print(f"Navigating to {target_url}...")
        page.goto(target_url, wait_until="networkidle")

        print("Waiting 15 seconds for manual login/session auth...")
        time.sleep(15)

        print("Switching Main Dashboard to MF CASH...")
        main_tab = page.locator("a:has-text('MF Cash'), li:has-text('MF Cash')").first
        if main_tab.is_visible():
            main_tab.click(force=True)
            time.sleep(2)

        print("Scanning for available months...")
        month_links = page.locator("table tbody tr td:first-child a").all()
        month_names = list(dict.fromkeys([l.inner_text().strip() for l in month_links if len(l.inner_text().strip()) > 4]))
        print(f"Found {len(month_names)} months to process.\n")

        current_year = None
        yearly_rows = []

        for month_name in month_names:
            year_str = month_name.split()[-1]
            year_int = int(year_str)

            if year_int < 2020:
                print("🛑 Reached 2019. Stopping fetch limit as requested.")
                if yearly_rows:
                    clean_and_save_data(yearly_rows, current_year)
                    yearly_rows = []
                break

            if current_year and current_year != year_str:
                clean_and_save_data(yearly_rows, current_year)
                yearly_rows = [] 

            current_year = year_str
            print(f"👉 Processing data for: {month_name}")
            monthly_rows = [] 
            
            try:
                # 1. Click the specific month link
                link_locator = page.locator(f"table tbody tr td:first-child a:has-text('{month_name}')").first
                link_locator.scroll_into_view_if_needed()
                link_locator.click(force=True)
                
                # 2. Wait for the modal to pop open
                modal_container = page.locator(".modal:visible").first
                modal_container.wait_for(state="visible", timeout=10000)
                time.sleep(1)

                # 3. Switch to the MF CASH tab inside the modal
                mf_cash_modal_tab = modal_container.locator("a:has-text('MF Cash'), li:has-text('MF Cash')").first
                if mf_cash_modal_tab.is_visible():
                    mf_cash_modal_tab.click(force=True)
                    
                    # 4. Wait for the VISIBLE MF headers to confirm the table swap is complete
                    modal_container.locator("div.dataTables_scrollHead table:visible th:has-text('MF EQUITY GROSS PURCHASE')").first.wait_for(state="visible", timeout=10000)
                    time.sleep(1.5)

                # 5. THE FIX: Target ONLY the ":visible" Body table, ignoring the hidden Summary table
                target_table = modal_container.locator("div.dataTables_scrollBody table:visible").first
                
                # Fallback if DataTables scroll structure isn't perfectly rendered
                if target_table.count() == 0:
                    target_table = modal_container.locator("table.dataTable:visible").last

                # 6. Extract and clean the rows
                for row in target_table.locator("tbody tr").all():
                    cells = row.locator("td").all()
                    if len(cells) >= 7:
                        date_text = cells[0].inner_text().strip()
                        
                        if any(k in date_text for k in ["Last", "Week", "Days", "Summary"]):
                            continue

                        if len(date_text) > 4:
                            row_data = [cell.inner_text().strip().replace(",", "") for cell in cells[:7]]
                            yearly_rows.append(row_data)
                            monthly_rows.append(row_data)
                
                print(f"   ✓ Extracted {len(monthly_rows)} records")
                if monthly_rows:
                    print(f"   [PREVIEW] {monthly_rows[0]}")

            except Exception as e:
                print(f"\n   ❌ ERROR ON {month_name}: Server took too long or modal got stuck.")
                traceback.print_exc()
                print("   Attempting auto-recovery...")
                page.reload(wait_until="networkidle")
                time.sleep(5)
                continue

            finally:
                # 7. Explicitly close and wait for modal to vanish
                try:
                    close_button = modal_container.locator("button.close").first
                    if close_button.is_visible():
                        close_button.click(force=True)
                    else:
                        page.keyboard.press("Escape")
                    
                    page.locator(".modal:visible").wait_for(state="hidden", timeout=10000)
                    time.sleep(0.5) 
                except:
                    print("   ⚠️ Modal failed to close properly. Forcing UI reset.")
                    page.reload(wait_until="networkidle")
                    time.sleep(5)
                
        if yearly_rows:
            clean_and_save_data(yearly_rows, current_year)

        browser.close()
        print("🎉 Complete!")


if __name__ == "__main__":
    scrape_mf_cash()