import pandas as pd
import numpy as np

def main():
    csv_path = "tmp/br_reed_bandit_rl.csv"
    qtable_path = "tmp/br_reed_bandit_rl_qtable.csv"

    print("=== RL Simulation Results Analysis ===\n")

    # 1. Packet Drops and Transmission Success Rate
    try:
        df = pd.read_csv(csv_path)
        total_drops = df['app_drops_interval'].sum()
        
        # Get the final row for each node to get total TX and RX counts
        final_stats = df.drop_duplicates(subset=['node_id'], keep='last')
        total_tx = final_stats['node_coap_tx_count'].sum()
        total_rx = final_stats['node_coap_rx_at_br_count'].sum()
        
        success_rate = (total_rx / total_tx * 100) if total_tx > 0 else 0
        
        print(f"[Overall Metrics]")
        print(f"Total Packet Drops: {total_drops}")
        print(f"Total Transmissions (TX): {total_tx}")
        print(f"Total Received at BR (RX): {total_rx}")
        print(f"Overall Transmission Success Rate: {success_rate:.2f}%\n")
        
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    # 2. Q-Table Results by Node Location
    try:
        q_df = pd.read_csv(qtable_path)
        
        # Group by node_name to find preferred action (weighted by action_count)
        # We can find the action with the maximum average Q-value per node, or maximum count
        
        node_stats = []
        for node_name in [f"reed{i}" for i in range(1, 26)]:
            node_q = q_df[q_df['node_name'] == node_name]
            if node_q.empty:
                continue
            
            # Action with the highest average Q-value
            avg_q = node_q.groupby(['action_index', 'action_adjustment_s'])['q_value'].mean().reset_index()
            best_action_q = avg_q.loc[avg_q['q_value'].idxmax()]
            
            # Action with the highest selection count (most exploited)
            total_counts = node_q.groupby(['action_index', 'action_adjustment_s'])['action_count'].sum().reset_index()
            most_used_action = total_counts.loc[total_counts['action_count'].idxmax()]
            
            node_stats.append({
                'node_name': node_name,
                'best_action_by_q': best_action_q['action_adjustment_s'],
                'most_used_action': most_used_action['action_adjustment_s'],
            })
            
        print("[Node-wise Q-Table Results (Location Analysis)]")
        print("Note: BR is at (40, 0).")
        print("Grid: row 0 (reed1-5) is closest to BR (y=20), row 4 (reed21-25) is furthest (y=100).\n")
        
        grid_most_used = []
        for r in range(5):
            row_str = []
            for c in range(5):
                idx = r * 5 + c
                if idx < len(node_stats):
                    row_str.append(f"{node_stats[idx]['most_used_action']:>4}")
                else:
                    row_str.append("N/A")
            grid_most_used.append(" | ".join(row_str))
            
        print("Most Exploited Action Adjustment (Multiplier) Grid:")
        for r, row_str in enumerate(grid_most_used):
            print(f"Row {r+1} (y={(r+1)*20:3}): [ {row_str} ]")

    except FileNotFoundError:
        print(f"File not found: {qtable_path}")

if __name__ == "__main__":
    main()
