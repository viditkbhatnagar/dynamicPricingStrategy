############################ streamlit_app.py ###################################
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

############################
# GLOBAL SETTINGS
############################
sns.set_theme(style='whitegrid')
np.random.seed(42)

# Basic price arms
price_arms = [5,6,7,8,9,10,11,12,13,14,15]
MAX_REVENUE = 15.0 * 15.0  # Approx upper bound for normalizing in bandits

# Enumerations for Q-learning states
demand_enum = {'Low':0,'Medium':1,'High':2}
weather_enum= {0:0, 1:1}
mood_enum   = {'Unhappy':0,'Neutral':1,'Happy':2}

############################
# 0) CORE ENVIRONMENT LOGIC
############################
def get_revenue_with_supply(chosen_price, demand, competitor_price, weather, mood, supply):
    """
    Approximates revenue given environment conditions.
    demand: 'Low','Medium','High' => 5,10,15 (requests)
    competitor_price: float
    weather: 0=clear,1=rain
    mood: 'Unhappy','Neutral','Happy'
    supply: int (# drivers available)
    """
    if chosen_price <= competitor_price:
        base_acceptance = 0.7
    else:
        base_acceptance = 0.4

    if demand == 'High':
        demand_factor = 15
    elif demand == 'Medium':
        demand_factor = 10
    else:
        demand_factor = 5

    weather_mult = 1.1 if weather==1 else 1.0
    mood_mult = {'Unhappy':0.8,'Neutral':1.0,'Happy':1.2}.get(mood,1.0)

    acceptance_raw = base_acceptance * weather_mult * mood_mult
    # incorporate supply vs. demand
    acceptance_supply = min(1.0, acceptance_raw * (supply / demand_factor))
    revenue = chosen_price * demand_factor * acceptance_supply
    return revenue

def update_mood(current_mood, chosen_price, competitor_price):
    """
    Simple rule-based mood transitions depending on price difference.
    """
    diff = chosen_price - competitor_price
    next_mood = current_mood
    if diff > 2:
        if current_mood == 'Happy':
            next_mood = 'Neutral'
        elif current_mood == 'Neutral':
            next_mood = 'Unhappy'
    elif diff < -2:
        if current_mood == 'Unhappy':
            next_mood = 'Neutral'
        elif current_mood == 'Neutral':
            next_mood = 'Happy'
    return next_mood

############################
# 1) DATA GENERATION
############################
def generate_data_original(T=300):
    """Original synthetic dataset."""
    df = pd.DataFrame({
        'timestep': range(T),
        'demand': np.random.choice(['Low','Medium','High'], size=T, p=[0.3,0.4,0.3]),
        'competitor_price': np.random.uniform(5,15,size=T),
        'weather': np.random.choice([0,1], size=T, p=[0.8,0.2]),
        'mood': np.random.choice(['Unhappy','Neutral','Happy'], size=T, p=[0.2,0.5,0.3]),
        'supply': np.random.randint(3,16,size=T)
    })
    return df

def generate_data_extended(T=300):
    """Extended dataset with different distribution."""
    df = pd.DataFrame({
        'timestep': range(T),
        'demand': np.random.choice(['Low','Medium','High'], size=T, p=[0.2,0.3,0.5]),
        'competitor_price': np.random.uniform(6,14,size=T),
        'weather': np.random.choice([0,1], size=T, p=[0.75,0.25]),
        'mood': np.random.choice(['Unhappy','Neutral','Happy'], size=T, p=[0.1,0.4,0.5]),
        'supply': np.random.randint(5,20,size=T)
    })
    return df

############################
# 2) BANDIT ALGORITHMS
############################
def run_bandits(df, method="both"):
    """
    Run Thompson Sampling & UCB on df, returning reward histories & means.
    method: "TS","UCB","both"
    """
    T = len(df)
    num_arms = len(price_arms)

    # For TS
    alpha = np.ones(num_arms)
    beta = np.ones(num_arms)
    ts_rewards = []
    ts_arms = []

    # For UCB
    ucb_counts = np.zeros(num_arms)
    ucb_sums   = np.zeros(num_arms)
    ucb_rewards= []
    ucb_arms   = []

    for t_i in range(T):
        row = df.iloc[t_i]

        # 1) Thompson
        if method in ["TS","both"]:
            thetas = np.random.beta(alpha, beta)
            arm_ts = np.argmax(thetas)
            price_ts = price_arms[arm_ts]
            rev_ts = get_revenue_with_supply(price_ts, row['demand'], row['competitor_price'],
                                             row['weather'], row['mood'], row['supply'])
            ts_rewards.append(rev_ts)
            ts_arms.append(arm_ts)

            # update
            r_norm = np.clip(rev_ts/MAX_REVENUE,0,1)
            alpha[arm_ts] += r_norm
            beta[arm_ts]  += (1-r_norm)

        # 2) UCB
        if method in ["UCB","both"]:
            ucb_values = []
            for i in range(num_arms):
                if ucb_counts[i]<1:
                    ucb_values.append(1e9)
                else:
                    avg_r = ucb_sums[i]/ucb_counts[i]
                    ub   = avg_r + np.sqrt(2*np.log(t_i+1)/(ucb_counts[i]))
                    ucb_values.append(ub)
            arm_ucb = np.argmax(ucb_values)
            price_ucb = price_arms[arm_ucb]
            rev_ucb = get_revenue_with_supply(price_ucb, row['demand'], row['competitor_price'],
                                              row['weather'], row['mood'], row['supply'])
            ucb_rewards.append(rev_ucb)
            ucb_arms.append(arm_ucb)

            # update
            ucb_counts[arm_ucb]+=1
            ucb_sums[arm_ucb]+=rev_ucb

    # compute average
    out = {}
    if method in ["TS","both"]:
        out["ts_rewards"] = ts_rewards
        out["ts_avg"] = np.mean(ts_rewards)
        out["ts_arms"] = ts_arms
    if method in ["UCB","both"]:
        out["ucb_rewards"] = ucb_rewards
        out["ucb_avg"] = np.mean(ucb_rewards)
        out["ucb_arms"] = ucb_arms

    return out

############################
# 3) Q-LEARNING (Simplified)
############################
def state_to_idx(row):
    d_i = demand_enum[row['demand']]
    w_i = weather_enum[row['weather']]
    m_i = mood_enum[row['mood']]
    return (d_i, w_i, m_i)

def run_qlearning(df, episodes=300, max_steps=20, alpha_q=0.1, gamma_q=0.9, epsilon=0.2):
    """
    Q-table shape = [demand(3) x weather(2) x mood(3) x actions(11)]
    We'll do random stepping for demonstration. 
    """
    n_d = 3
    n_w = 2
    n_m = 3
    n_a = len(price_arms)
    Q = np.zeros((n_d,n_w,n_m,n_a))

    for ep in range(episodes):
        # pick random row as start
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]
        s_tuple = state_to_idx(row)

        for step in range(max_steps):
            # epsilon-greedy
            if np.random.rand()<epsilon:
                a_idx = np.random.randint(0,n_a)
            else:
                a_idx = np.argmax(Q[s_tuple])

            chosen_price = price_arms[a_idx]
            reward = get_revenue_with_supply(chosen_price, row['demand'], row['competitor_price'],
                                             row['weather'], row['mood'], row['supply'])

            new_mood = update_mood(row['mood'], chosen_price, row['competitor_price'])
            # random next row
            next_idx = np.random.randint(0, len(df))
            next_row = df.iloc[next_idx].copy()
            next_row['mood'] = new_mood
            s2_tuple = state_to_idx(next_row)

            best_next_a = np.argmax(Q[s2_tuple])
            old_q = Q[s_tuple+(a_idx,)]
            Q[s_tuple+(a_idx,)] = old_q + alpha_q*(reward + gamma_q*Q[s2_tuple+(best_next_a,)] - old_q)

            # next
            row = next_row
            s_tuple = s2_tuple

    return Q

############################
# 4) MCDM (TOPSIS) EXAMPLE
############################
def mcdm_criteria_for_price(p, df, n_samples=200):
    """Simulate average revenue, competitor diff, satisfaction for a price p."""
    rev_list = []
    diff_list= []
    sat_list = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]
        r = get_revenue_with_supply(p, row['demand'], row['competitor_price'],
                                    row['weather'], row['mood'], row['supply'])
        rev_list.append(r)
        diff_list.append(abs(p - row['competitor_price']))
        sat = 4.5 if p < row['competitor_price'] else 3.2
        sat_list.append(sat)
    return np.mean(rev_list), np.mean(diff_list), np.mean(sat_list)

def run_topsis(prices, df):
    """
    Basic TOPSIS for [Revenue, Diff, Satisfaction].
    weights = [0.5, 0.2, 0.3] (example).
    """
    w = np.array([0.5,0.2,0.3])
    mat = []
    for p in prices:
        rev, diff, sat = mcdm_criteria_for_price(p, df)
        mat.append([rev, diff, sat])
    mat = np.array(mat)

    # Normalize
    norms = np.sqrt((mat**2).sum(axis=0))
    norm_mat = mat / norms

    # Weighted
    weighted = norm_mat * w

    # Ideal best/worst
    # col0,2 => max=best, col1 => min=best
    ideal_best  = np.array([weighted[:,0].max(), weighted[:,1].min(), weighted[:,2].max()])
    ideal_worst = np.array([weighted[:,0].min(), weighted[:,1].max(), weighted[:,2].min()])

    dist_best = np.sqrt(((weighted-ideal_best)**2).sum(axis=1))
    dist_worst= np.sqrt(((weighted-ideal_worst)**2).sum(axis=1))

    scores = dist_worst/(dist_best+dist_worst)
    results = sorted(zip(prices, scores), key=lambda x:x[1], reverse=True)
    return results, mat

############################
# STREAMLIT APP
############################
def main():
    st.title("Dynamic Pricing for Ride-Sharing: Enhanced Streamlit App")
    st.markdown("""
    This **all-in-one** app demonstrates:
    1. **Data Generation** (Original vs. Extended)
    2. **Bandit Algorithms** (Thompson & UCB) with advanced visualizations
    3. **Q-learning** on discrete states (demand, weather, mood)
    4. **TOPSIS** for Multi-Criteria Decision Making
    5. **In-app** dataset preview & download
    ---
    """)

    #############################################
    # SIDEBAR CONTROLS
    #############################################
    st.sidebar.header("1) Choose Dataset")
    dataset_choice = st.sidebar.selectbox("Dataset:", ["Original","Extended"])
    if dataset_choice=="Original":
        df = generate_data_original(300)
    else:
        df = generate_data_extended(300)

    st.sidebar.header("2) Bandit Parameters")
    episodes_bandit = st.sidebar.slider("Timesteps for Bandits", 50, 1000, 300, step=50)

    st.sidebar.header("3) Q-Learning Parameters")
    q_episodes = st.sidebar.slider("Q-learning Episodes", 50, 1000, 300, step=50)
    q_steps    = st.sidebar.slider("Max Steps Per Episode", 5, 50, 20, step=5)
    alpha_q    = st.sidebar.slider("Q-learning alpha", 0.01, 1.0, 0.1, step=0.01)
    gamma_q    = st.sidebar.slider("Q-learning gamma", 0.01, 1.0, 0.9, step=0.01)
    epsilon_q  = st.sidebar.slider("Q-learning epsilon", 0.01, 1.0, 0.2, step=0.01)

    st.sidebar.header("4) MCDM (TOPSIS)")
    run_mcdm_flag = st.sidebar.checkbox("Run TOPSIS?", value=False)
    candidate_prices_str = st.sidebar.text_input("Candidate Prices (comma-separated)", "7,9,11,13")

    # Apply timesteps for bandit
    if len(df) > episodes_bandit:
        df = df.iloc[:episodes_bandit].copy()
    else:
        # If generated data smaller than episodes, we replicate
        repeat_factor = int(np.ceil(episodes_bandit/len(df)))
        df = pd.concat([df]*repeat_factor, ignore_index=True)
        df = df.iloc[:episodes_bandit].copy()

    #############################################
    # SHOW DATASET
    #############################################
    st.subheader("Dataset Preview & Download")
    st.write(df.head(10))

    # Download button
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Current Dataset as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{dataset_choice}_dataset.csv",
        mime="text/csv"
    )

    #############################################
    # RUN BANDIT ANALYSIS
    #############################################
    st.subheader("Bandit Analysis (Thompson & UCB)")
    if st.button("Run Bandit Analysis"):
        results = run_bandits(df, method="both")
        ts_rewards = results["ts_rewards"]
        ucb_rewards= results["ucb_rewards"]
        ts_avg     = results["ts_avg"]
        ucb_avg    = results["ucb_avg"]
        ts_arms    = results["ts_arms"]
        ucb_arms   = results["ucb_arms"]

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Thompson Sampling Avg.** = {ts_avg:.2f}")
            fig_ts, ax_ts = plt.subplots()
            ax_ts.plot(ts_rewards, label="TS Rewards")
            ax_ts.set_title("Thompson Sampling - Rewards Over Time")
            ax_ts.set_xlabel("Timestep")
            ax_ts.set_ylabel("Revenue")
            st.pyplot(fig_ts)

            # Confidence intervals or rolling means
            window=20
            rolling_means=[]
            rolling_stds=[]
            steps=[]
            for i in range(0,len(ts_rewards)-window,window):
                chunk = ts_rewards[i:i+window]
                rolling_means.append(np.mean(chunk))
                rolling_stds.append(np.std(chunk))
                steps.append(i+window//2)
            fig_ts2, ax_ts2 = plt.subplots()
            ax_ts2.plot(steps, rolling_means, label="Rolling Mean")
            ax_ts2.fill_between(steps,
                                np.array(rolling_means)-np.array(rolling_stds),
                                np.array(rolling_means)+np.array(rolling_stds),
                                alpha=0.2, label="±1 std")
            ax_ts2.set_title("Thompson Sampling Rolling Mean & Std")
            ax_ts2.legend()
            st.pyplot(fig_ts2)

        with col2:
            st.write(f"**UCB Avg.** = {ucb_avg:.2f}")
            fig_ucb, ax_ucb = plt.subplots()
            ax_ucb.plot(ucb_rewards, color='orange')
            ax_ucb.set_title("UCB - Rewards Over Time")
            ax_ucb.set_xlabel("Timestep")
            ax_ucb.set_ylabel("Revenue")
            st.pyplot(fig_ucb)

            # Distribution of chosen arms
            fig_armdist, ax_armdist = plt.subplots()
            sns.countplot(x=ts_arms, color='blue', alpha=0.5, label="TS", ax=ax_armdist)
            sns.countplot(x=ucb_arms, color='red', alpha=0.5, label="UCB", ax=ax_armdist)
            ax_armdist.set_title("Distribution of Arm Selections (TS=Blue, UCB=Red)")
            ax_armdist.set_xlabel("Arm Index (price_arms)")
            ax_armdist.legend()
            st.pyplot(fig_armdist)

    #############################################
    # RUN Q-LEARNING
    #############################################
    st.subheader("Q-learning (Discrete States: Demand, Weather, Mood)")
    if st.button("Run Q-learning"):
        Q = run_qlearning(df, episodes=q_episodes, max_steps=q_steps,
                          alpha_q=alpha_q, gamma_q=gamma_q, epsilon=epsilon_q)
        st.write("**Q-learning completed.**")
        st.write("Q-table shape:", Q.shape)
        st.markdown("""
        You can interpret the Q-table by:
        - picking a state (demand, weather, mood)
        - finding the argmax over the 11 actions (price_arms).
        """)

        # We'll show a small slice: e.g., weather=0 => slice Q to get best action
        st.write("### Heatmap: Best Action for demand x mood (weather=0)")

        # best action for each (demand, mood) at weather=0
        dm_best = np.zeros((3,3))  # 3 demands, 3 moods
        for d_i in [0,1,2]:
            for m_i in [0,1,2]:
                sliceQ = Q[d_i,0,m_i,:]  # weather=0
                best_a = np.argmax(sliceQ)
                dm_best[d_i,m_i] = price_arms[best_a]

        fig_hm, ax_hm = plt.subplots()
        sns.heatmap(dm_best, annot=True, fmt=".0f", cmap="YlGnBu",
                    xticklabels=['Unhappy','Neutral','Happy'],
                    yticklabels=['Low','Medium','High'], ax=ax_hm)
        ax_hm.set_title("Best Price (Action) @ Weather=0 via Q-table")
        ax_hm.set_xlabel("Mood")
        ax_hm.set_ylabel("Demand")
        st.pyplot(fig_hm)

    #############################################
    # RUN TOPSIS MCDM
    #############################################
    if run_mcdm_flag:
        st.subheader("TOPSIS for Price Selection")
        c_prices = candidate_prices_str.strip()
        if c_prices:
            try:
                cand_prices = [int(x) for x in c_prices.split(",")]
            except:
                st.error("Invalid candidate prices input.")
                cand_prices = []
        else:
            cand_prices = [7,9,11,13]

        if len(cand_prices)>0:
            results, mat = run_topsis(cand_prices, df)
            st.write("**TOPSIS Ranking (descending)**:")
            for p,score in results:
                st.write(f"Price={p} => Score={score:.4f}")

            st.write("**Raw Criteria Matrix** [Revenue, Diff, Satisfaction]")
            st.dataframe(pd.DataFrame(mat, columns=["Revenue","Diff","Satisfaction"], index=cand_prices))

    st.markdown("---")
    st.markdown("""
    ### Conclusion & Next Steps
    - **Bandits** (TS/UCB) can adapt prices in real-time, balancing exploration & exploitation.
    - **Q-learning** handles **state-based** (demand/weather/mood) decisions.
    - **TOPSIS** offers a multi-criteria approach factoring in revenue, competitor gap, satisfaction.
    - This app unifies everything into one interactive **Streamlit** interface.
    """)

if __name__=="__main__":
    main()
