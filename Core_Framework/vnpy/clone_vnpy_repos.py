#!/usr/bin/env python3
"""
Script to clone all vnpy repositories from GitHub
"""

import json
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Repository data from GitHub API
repos_data = [
    {"name": "code_demo", "clone_url": "https://github.com/vnpy/code_demo.git"},
    {"name": "FinRL", "clone_url": "https://github.com/vnpy/FinRL.git"},
    {"name": "flaskbb", "clone_url": "https://github.com/vnpy/flaskbb.git"},
    {"name": "gplearn", "clone_url": "https://github.com/vnpy/gplearn.git"},
    {"name": "hashkey-pro-api", "clone_url": "https://github.com/vnpy/hashkey-pro-api.git"},
    {"name": "Kronos", "clone_url": "https://github.com/vnpy/Kronos.git"},
    {"name": "new_demo", "clone_url": "https://github.com/vnpy/new_demo.git"},
    {"name": "vnag", "clone_url": "https://github.com/vnpy/vnag.git"},
    {"name": "vnpy", "clone_url": "https://github.com/vnpy/vnpy.git"},
    {"name": "vnpy_algotrading", "clone_url": "https://github.com/vnpy/vnpy_algotrading.git"},
    {"name": "vnpy_alpaca", "clone_url": "https://github.com/vnpy/vnpy_alpaca.git"},
    {"name": "vnpy_arctic", "clone_url": "https://github.com/vnpy/vnpy_arctic.git"},
    {"name": "vnpy_bingx", "clone_url": "https://github.com/vnpy/vnpy_bingx.git"},
    {"name": "vnpy_binance", "clone_url": "https://github.com/vnpy/vnpy_binance.git"},
    {"name": "vnpy_bitget", "clone_url": "https://github.com/vnpy/vnpy_bitget.git"},
    {"name": "vnpy_bitmex", "clone_url": "https://github.com/vnpy/vnpy_bitmex.git"},
    {"name": "vnpy_bybit", "clone_url": "https://github.com/vnpy/vnpy_bybit.git"},
    {"name": "vnpy_coinbase", "clone_url": "https://github.com/vnpy/vnpy_coinbase.git"},
    {"name": "vnpy_ctastrategy", "clone_url": "https://github.com/vnpy/vnpy_ctastrategy.git"},
    {"name": "vnpy_ctp", "clone_url": "https://github.com/vnpy/vnpy_ctp.git"},
    {"name": "vnpy_ctptest", "clone_url": "https://github.com/vnpy/vnpy_ctptest.git"},
    {"name": "vnpy_da", "clone_url": "https://github.com/vnpy/vnpy_da.git"},
    {"name": "vnpy_datarecorder", "clone_url": "https://github.com/vnpy/vnpy_datarecorder.git"},
    {"name": "vnpy_deribit", "clone_url": "https://github.com/vnpy/vnpy_deribit.git"},
    {"name": "vnpy_dolphindb", "clone_url": "https://github.com/vnpy/vnpy_dolphindb.git"},
    {"name": "vnpy_esunny", "clone_url": "https://github.com/vnpy/vnpy_esunny.git"},
    {"name": "vnpy_femas", "clone_url": "https://github.com/vnpy/vnpy_femas.git"},
    {"name": "vnpy_gateio", "clone_url": "https://github.com/vnpy/vnpy_gateio.git"},
    {"name": "vnpy_gm", "clone_url": "https://github.com/vnpy/vnpy_gm.git"},
    {"name": "vnpy_hashkey", "clone_url": "https://github.com/vnpy/vnpy_hashkey.git"},
    {"name": "vnpy_hts", "clone_url": "https://github.com/vnpy/vnpy_hts.git"},
    {"name": "vnpy_huobi", "clone_url": "https://github.com/vnpy/vnpy_huobi.git"},
    {"name": "vnpy_ib", "clone_url": "https://github.com/vnpy/vnpy_ib.git"},
    {"name": "vnpy_influxdb", "clone_url": "https://github.com/vnpy/vnpy_influxdb.git"},
    {"name": "vnpy_jees", "clone_url": "https://github.com/vnpy/vnpy_jees.git"},
    {"name": "vnpy_ksgold", "clone_url": "https://github.com/vnpy/vnpy_ksgold.git"},
    {"name": "vnpy_lstar", "clone_url": "https://github.com/vnpy/vnpy_lstar.git"},
    {"name": "vnpy_mcdata", "clone_url": "https://github.com/vnpy/vnpy_mcdata.git"},
    {"name": "vnpy_mini", "clone_url": "https://github.com/vnpy/vnpy_mini.git"},
    {"name": "vnpy_mysql", "clone_url": "https://github.com/vnpy/vnpy_mysql.git"},
    {"name": "vnpy_okx", "clone_url": "https://github.com/vnpy/vnpy_okx.git"},
    {"name": "vnpy_optionmaster", "clone_url": "https://github.com/vnpy/vnpy_optionmaster.git"},
    {"name": "vnpy_papertrading", "clone_url": "https://github.com/vnpy/vnpy_papertrading.git"},
    {"name": "vnpy_portfoliomanager", "clone_url": "https://github.com/vnpy/vnpy_portfoliomanager.git"},
    {"name": "vnpy_portfoliostrategy", "clone_url": "https://github.com/vnpy/vnpy_portfoliostrategy.git"},
    {"name": "vnpy_postgresql", "clone_url": "https://github.com/vnpy/vnpy_postgresql.git"},
    {"name": "vnpy_rest", "clone_url": "https://github.com/vnpy/vnpy_rest.git"},
    {"name": "vnpy_rohon", "clone_url": "https://github.com/vnpy/vnpy_rohon.git"},
    {"name": "vnpy_rpcservice", "clone_url": "https://github.com/vnpy/vnpy_rpcservice.git"},
    {"name": "vnpy_rqdata", "clone_url": "https://github.com/vnpy/vnpy_rqdata.git"},
    {"name": "vnpy_scripttrader", "clone_url": "https://github.com/vnpy/vnpy_scripttrader.git"},
    {"name": "vnpy_sqlite", "clone_url": "https://github.com/vnpy/vnpy_sqlite.git"},
    {"name": "vnpy_spreadtrading", "clone_url": "https://github.com/vnpy/vnpy_spreadtrading.git"},
    {"name": "vnpy_tap", "clone_url": "https://github.com/vnpy/vnpy_tap.git"},
    {"name": "vnpy_tora", "clone_url": "https://github.com/vnpy/vnpy_tora.git"},
    {"name": "vnpy_tts", "clone_url": "https://github.com/vnpy/vnpy_tts.git"},
    {"name": "vnpy_tushare", "clone_url": "https://github.com/vnpy/vnpy_tushare.git"},
    {"name": "vnpy_uft", "clone_url": "https://github.com/vnpy/vnpy_uft.git"},
    {"name": "vnpy_ust", "clone_url": "https://github.com/vnpy/vnpy_ust.git"},
    {"name": "vnpy_voltrader", "clone_url": "https://github.com/vnpy/vnpy_voltrader.git"},
    {"name": "vnpy_websocket", "clone_url": "https://github.com/vnpy/vnpy_websocket.git"},
    {"name": "vnpy_webtrader", "clone_url": "https://github.com/vnpy/vnpy_webtrader.git"},
    {"name": "vnpy_wind", "clone_url": "https://github.com/vnpy/vnpy_wind.git"},
    {"name": "vnpy_xt", "clone_url": "https://github.com/vnpy/vnpy_xt.git"},
    {"name": "vnpy_xtp", "clone_url": "https://github.com/vnpy/vnpy_xtp.git"}
]

def clone_repository(repo_info, target_dir):
    """Clone a single repository"""
    repo_name = repo_info["name"]
    clone_url = repo_info["clone_url"]
    repo_path = os.path.join(target_dir, repo_name)
    
    print(f"Cloning {repo_name}...")
    
    try:
        # Check if directory already exists
        if os.path.exists(repo_path):
            print(f"  {repo_name} already exists, skipping...")
            return f"SKIPPED: {repo_name}"
        
        # Clone the repository
        result = subprocess.run(
            ["git", "clone", clone_url, repo_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per repo
        )
        
        if result.returncode == 0:
            print(f"  ✓ Successfully cloned {repo_name}")
            return f"SUCCESS: {repo_name}"
        else:
            print(f"  ✗ Failed to clone {repo_name}: {result.stderr}")
            return f"FAILED: {repo_name} - {result.stderr}"
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout cloning {repo_name}")
        return f"TIMEOUT: {repo_name}"
    except Exception as e:
        print(f"  ✗ Error cloning {repo_name}: {str(e)}")
        return f"ERROR: {repo_name} - {str(e)}"

def main():
    target_dir = "/Users/hacker/PycharmProjects/vnpy_freedom"
    
    print(f"Starting to clone {len(repos_data)} repositories to {target_dir}")
    print("=" * 60)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Clone repositories with threading for faster execution
    max_workers = 5  # Limit concurrent clones to avoid overwhelming the system
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clone tasks
        future_to_repo = {
            executor.submit(clone_repository, repo, target_dir): repo 
            for repo in repos_data
        }
        
        # Process completed tasks
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                error_msg = f"ERROR: {repo['name']} generated an exception: {exc}"
                print(error_msg)
                results.append(error_msg)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLONE SUMMARY:")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    skipped_count = sum(1 for r in results if r.startswith("SKIPPED"))
    failed_count = len(results) - success_count - skipped_count
    
    print(f"Total repositories: {len(repos_data)}")
    print(f"Successfully cloned: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed repositories:")
        for result in results:
            if not result.startswith("SUCCESS") and not result.startswith("SKIPPED"):
                print(f"  {result}")
    
    print(f"\nAll repositories are now available in: {target_dir}")

if __name__ == "__main__":
    main()
