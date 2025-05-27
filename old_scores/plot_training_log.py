import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def plot_training_data_separated_with_avg(csv_filepath, output_dir="training_plots_avg"):
    if not os.path.exists(csv_filepath):
        print(f"Error: File {csv_filepath} not found.")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if df.empty:
        print("CSV file is empty. Cannot generate plots.")
        return

    numeric_cols = ['Episode', 'Score', 'Epsilon', 'Steps', 'TotalReward']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Episode', 'Score'], inplace=True)

    if df.empty:
        print("No valid numeric data for 'Episode' and 'Score'. Cannot generate plots.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Dynamic calculations
    df['Calculated_MaxScore'] = df['Score'].cummax()
    df['Calculated_AvgScore100'] = df['Score'].rolling(window=100, min_periods=1).mean()
    df['Calculated_AvgScore50'] = df['Score'].rolling(window=50, min_periods=1).mean()

    if 'TotalReward' in df.columns:
        df['AvgTotalReward100'] = df['TotalReward'].rolling(window=100, min_periods=1).mean()
    if 'Steps' in df.columns:
        df['AvgSteps100'] = df['Steps'].rolling(window=100, min_periods=1).mean()

    # Plot: Score, MaxScore, AvgScore100
    if 'Score' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.plot(df['Episode'], df['Score'], label='Score per Episode', alpha=0.5, color='skyblue')
        plt.plot(df['Episode'], df['Calculated_MaxScore'], label='Max Score', color='darkblue', linestyle='--')
        plt.plot(df['Episode'], df['Calculated_AvgScore100'], label='Avg Score (100 episodes)', color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Score, Max Score, and Moving Average (100 episodes)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_score_max_avg.png"))
        plt.close()

    # Plot: AvgScore50
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Calculated_AvgScore50'], label='Avg Score (50 episodes)', color='teal')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('Moving Average Score (50 episodes)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_avg_score_50.png"))
    plt.close()

    # Plot: Epsilon
    if 'Epsilon' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Episode'], df['Epsilon'], label='Epsilon Value', color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_epsilon.png"))
        plt.close()

    # Plot: TotalReward and AvgTotalReward100
    if 'TotalReward' in df.columns and 'AvgTotalReward100' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.plot(df['Episode'], df['TotalReward'], label='Total Reward per Episode', alpha=0.5, color='lightcoral')
        plt.plot(df['Episode'], df['AvgTotalReward100'], label='Avg Total Reward (100 episodes)', color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward and Moving Average (100 episodes)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_total_reward_avg.png"))
        plt.close()

    # Plot: Steps and AvgSteps100
    if 'Steps' in df.columns and 'AvgSteps100' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.plot(df['Episode'], df['Steps'], label='Steps per Episode', alpha=0.5, color='navajowhite')
        plt.plot(df['Episode'], df['AvgSteps100'], label='Avg Steps (100 episodes)', color='orange', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode and Moving Average (100 episodes)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_steps_avg.png"))
        plt.close()

    print(f"All plots saved in directory: {output_dir}")


if __name__ == '__main__':
    csv_file = "snake_dqn_training_log2_01.csv"
    output_directory = "training_plots_with_avg"
    plot_training_data_separated_with_avg(csv_file, output_dir=output_directory)