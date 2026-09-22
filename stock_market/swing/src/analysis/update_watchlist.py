import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STOCKS_FILE = BASE_DIR / "Stocks.txt"

NEW_ENTRIES = [
    "Stocks to Watch Today: Senco Gold, HDFC Bank, BEL, Bharat Dynamics, Nykaa in focus on 06 July",
    "Stocks to Watch Today: Infosys, Coal India, Hexaware Technologies in focus on 02 July",
    "Stocks to Watch Today: Hexaware Technologies, Godrej Properties, BPCL, Tata Communications, Max Healthcare, Maruti Suzuki, ITC, Hindustan Unilever, SBI, Dixon Technologies, Bharti Airtel, Sun Pharma, Kotak Mahindra Bank, Exide Industries in focus on 01 July",
    "Stocks to Watch Today: HDFC Bank, SJVN, YES Bank, Bandhan Bank, Dr. Reddy's, Max Healthcare, Lupin, BEL, Coal India, Trent, Bharat Forge, ICICI Bank, JSW Energy, BHEL in focus on 30 June",
    "Stocks to Watch Today: Kotak Mahindra Bank, Torrent Power, Persistent Systems, Aurobindo Pharma, Lupin, Dr. Reddy's Laboratories, BEML, Zydus Lifesciences, Reliance Industries in focus on 29 June",
    "Stocks to Watch Today: Trent in focus on 26 June",
    "Stocks to Watch Today: Olectra Greentech, GAIL, Hindustan Aeronautics, Bharat Forge, Vedanta, Equitas Small Finance Bank in focus on 25 June",
    "Stocks to Watch Today: Eicher Motors, Torrent Power, Bajaj Auto, Vodafone Idea, YES Bank, Infosys, Wipro, Honasa Consumer, IRFC, Tata Power in focus on 24 June",
    "Stocks to Watch Today: Tech Mahindra, Sun Pharma, Reliance Industries, Infosys, Bharat Electronics, Gabriel India in focus on 23 June",
    "Stocks to Watch Today: SBI, Bharat Electronics, Tata Motors, Voltas, South Indian Bank, Reliance Industries, Alembic Pharmaceuticals, HDFC Bank, Sun Pharma, Bajaj Finance in focus on 22 June",
    "Stocks to Watch Today: Reliance Industries, Infosys, Wipro, Bharat Electronics, Divis Labs, Adani Enterprises, HCL Technologies, Bharat Forge in focus on 19 June",
    "Stocks to Watch Today: RVNL, HFCL, Bosch, Lupin, Apollo Hospitals, Balkrishna Industries, Max Healthcare, Bharat Electronics in focus on 18 June",
    "Stocks to Watch Today: Tata Motors, Wipro, General Insurance Corporation, Infosys, Siemens, Mahanagar Gas, Trent in focus on 17 June",
    "Stocks to Watch Today: HCL Technologies, SBI, Adani Enterprises, IRCTC, GIC, Reliance Industries, BPCL, Mahanagar Gas in focus on 16 June",
    "Stocks to Watch Today: Karnataka Bank, Avalon Technologies, Vedanta, JSW Energy, Aries Agro, Vodafone Idea, Yes Bank in focus on 15 June",
    "Stocks to Watch Today: HDFC Bank, BPCL, IndiGo, Vedanta, SBI, Dabur, Happiest Minds, Astral in focus on 12 June",
    "Stocks to Watch Today: ONGC, Max Financial, ICICI Bank, HUL, ITC, Infosys, HCL Tech, Canara Bank, Bank of Baroda, Zee Entertainment in focus on 11 June",
]

def main():
    if not STOCKS_FILE.exists():
        print(f"Error: {STOCKS_FILE} does not exist.")
        return

    with open(STOCKS_FILE, "r", encoding="utf-8") as fh:
        current_content = fh.read().strip()

    # Create list of lines, filtering out empty lines
    current_lines = [line.strip() for line in current_content.split("\n") if line.strip()]

    # We prepend the new entries so that the newest dates are at the top,
    # matching the descending date order style of the existing file.
    all_lines = NEW_ENTRIES + current_lines

    with open(STOCKS_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(all_lines) + "\n")

    print(f"Successfully updated {STOCKS_FILE} with {len(NEW_ENTRIES)} new watchlists.")

if __name__ == "__main__":
    main()
