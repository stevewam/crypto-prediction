        if t != 1:
            sharpe_select_df = []

            coins_stats_df['avg_expected_roi'] = coins_stats_df['expected_roi'].expanding().mean()
            coins_stats_df['new_mean'] = coins_stats_df['avg_expected_roi'].apply(lambda x: update_mean(mean_roi, t, x))
            coins_stats_df['new_std'] = coins_stats_df.apply(lambda row: update_std(std_roi, mean_roi, row['new_mean'], t, row['avg_expected_roi']), axis=1)
            coins_stats_df['sharpe_ratio'] = coins_stats_df['new_mean']/coins_stats_df['new_std']
            coins_stats_df['n'] = np.arange(start=1, stop=(len(coins_stats_df)+1))

            n = coins_stats_df[coins_stats_df['sharpe_ratio']==coins_stats_df['sharpe_ratio'].max()]['n'].values[0]
            
            top_roi = coins_stats_df.iloc[:n,:]['expected_roi'].values

        else:
            n = initial_n