import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from director_alpha import config

def generate_dummy_data():
    print("Generating dummy data...")
    
    # 1. Compustat
    # gvkey, datadate, fyear, fic, at, oibdp, prcc_f, csho, ceq, dltt, dlc, xrd, capx, sich, naics
    n_firms = 50
    years = range(2000, 2010)
    data = []
    for gvkey in range(1000, 1000 + n_firms):
        for year in years:
            data.append({
                'gvkey': str(gvkey),
                'datadate': pd.Timestamp(f"{year}-12-31"),
                'fyear': year,
                'fic': 'USA',
                'at': np.random.uniform(100, 1000),
                'oibdp': np.random.uniform(10, 100),
                'prcc_f': np.random.uniform(10, 50),
                'csho': np.random.uniform(10, 20),
                'ceq': np.random.uniform(50, 500),
                'dltt': np.random.uniform(10, 100),
                'dlc': np.random.uniform(5, 50),
                'xrd': np.random.uniform(0, 10),
                'capx': np.random.uniform(5, 20),
                'sich': 3571, # Tech
                'naics': '334111'
            })
    compustat = pd.DataFrame(data)
    compustat.to_parquet(config.RAW_COMPUSTAT_PATH)
    
    # 2. CRSP
    # permno, date, shrcd, siccd, prc, ret, ncusip
    crsp_data = []
    for gvkey in range(1000, 1000 + n_firms):
        permno = gvkey + 50000
        for year in years:
            # Annual entry for simplicity in Phase 0 check
            crsp_data.append({
                'permno': permno,
                'date': pd.Timestamp(f"{year}-12-31"),
                'shrcd': 11,
                'siccd': 3571,
                'prc': np.random.uniform(10, 50),
                'ret': np.random.uniform(-0.1, 0.2),
                'ncusip': f"{gvkey}10" # Fake CUSIP
            })
    crsp = pd.DataFrame(crsp_data)
    crsp.to_parquet(config.RAW_CRSP_PATH)
    
    # 3. CCM
    # gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
    ccm_data = []
    for gvkey in range(1000, 1000 + n_firms):
        ccm_data.append({
            'gvkey': str(gvkey),
            'lpermno': gvkey + 50000,
            'linkdt': pd.Timestamp("1990-01-01"),
            'linkenddt': pd.Timestamp("2020-12-31"),
            'linktype': 'LU',
            'linkprim': 'P'
        })
    ccm = pd.DataFrame(ccm_data)
    ccm.to_parquet(config.RAW_CCM_PATH)
    
    # 4. ExecuComp
    # gvkey, year, execid, pceo, ceoann, becameceo, leftofc, joined_co, title, tdc1, age, gender
    exec_data = []
    for gvkey in range(1000, 1000 + n_firms):
        # CEO 1: 2000-2005
        exec_data.append({
            'gvkey': str(gvkey),
            'year': 2000,
            'execid': f"E{gvkey}A",
            'pceo': 'CEO',
            'ceoann': 'CEO',
            'becameceo': pd.Timestamp("2000-01-01"),
            'leftofc': pd.Timestamp("2005-12-31"),
            'joined_co': pd.Timestamp("1995-01-01"),
            'title': 'CEO',
            'tdc1': 1000,
            'age': 50,
            'gender': 'M'
        })
        # Add more years for CEO 1
        for y in range(2001, 2006):
            exec_data.append({
                'gvkey': str(gvkey),
                'year': y,
                'execid': f"E{gvkey}A",
                'pceo': 'CEO',
                'ceoann': 'CEO',
                'becameceo': pd.Timestamp("2000-01-01"),
                'leftofc': pd.Timestamp("2005-12-31"),
                'joined_co': pd.Timestamp("1995-01-01"),
                'title': 'CEO',
                'tdc1': 1000,
                'age': 50 + (y-2000),
                'gender': 'M'
            })
            
        # CEO 2: 2006-2010
        exec_data.append({
            'gvkey': str(gvkey),
            'year': 2006,
            'execid': f"E{gvkey}B",
            'pceo': 'CEO',
            'ceoann': 'CEO',
            'becameceo': pd.Timestamp("2006-01-01"),
            'leftofc': pd.NaT,
            'joined_co': pd.Timestamp("2006-01-01"), # External
            'title': 'CEO',
            'tdc1': 1200,
            'age': 45,
            'gender': 'F'
        })
        for y in range(2007, 2011):
            exec_data.append({
                'gvkey': str(gvkey),
                'year': y,
                'execid': f"E{gvkey}B",
                'pceo': 'CEO',
                'ceoann': 'CEO',
                'becameceo': pd.Timestamp("2006-01-01"),
                'leftofc': pd.NaT,
                'joined_co': pd.Timestamp("2006-01-01"),
                'title': 'CEO',
                'tdc1': 1200,
                'age': 45 + (y-2006),
                'gender': 'F'
            })
            
    execucomp = pd.DataFrame(exec_data)
    execucomp.to_parquet(config.RAW_EXECUCOMP_PATH)
    
    # 5. BoardEx Data
    # Directors: company_id, director_id, date_start, date_end, role_name
    # Committees: company_id, director_id, committee_name, c_date_start, c_date_end
    # Link: gvkey, company_id
    
    # Link Table
    link_data = []
    for gvkey in range(1000, 1000 + n_firms):
        link_data.append({
            'gvkey': str(gvkey),
            'company_id': f"C{gvkey}"
        })
    link = pd.DataFrame(link_data)
    link.to_parquet(config.RAW_BOARDEX_LINK_PATH)
    
    # Directors
    bx_directors_data = []
    bx_committees_data = []
    
    for gvkey in range(1000, 1000 + n_firms):
        company_id = f"C{gvkey}"
        # 5 directors per firm
        for i in range(5):
            director_id = f"D{gvkey}_{i}"
            # Active from 2000 to 2010
            bx_directors_data.append({
                'company_id': company_id,
                'director_id': director_id,
                'date_start': pd.Timestamp("2000-01-01"),
                'date_end': pd.Timestamp("2010-12-31"),
                'role_name': 'Director'
            })
            
            # Committees
            # Director 0 is Nom/Gov
            if i == 0:
                bx_committees_data.append({
                    'company_id': company_id,
                    'director_id': director_id,
                    'committee_name': 'Nomination & Governance',
                    'c_date_start': pd.Timestamp("2000-01-01"),
                    'c_date_end': pd.Timestamp("2010-12-31")
                })
            else:
                bx_committees_data.append({
                    'company_id': company_id,
                    'director_id': director_id,
                    'committee_name': 'Audit',
                    'c_date_start': pd.Timestamp("2000-01-01"),
                    'c_date_end': pd.Timestamp("2010-12-31")
                })
                
    # Add connectivity: D1000_0 also on C1001
    bx_directors_data.append({
        'company_id': "C1001",
        'director_id': "D1000_0",
        'date_start': pd.Timestamp("2002-01-01"),
        'date_end': pd.Timestamp("2010-12-31"),
        'role_name': 'Director'
    })
    
    directors = pd.DataFrame(bx_directors_data)
    directors.to_parquet(config.RAW_BOARDEX_DIRECTORS_PATH)
    
    committees = pd.DataFrame(bx_committees_data)
    committees.to_parquet(config.RAW_BOARDEX_COMMITTEES_PATH)
    
    print("Dummy data generated successfully.")

if __name__ == "__main__":
    generate_dummy_data()
