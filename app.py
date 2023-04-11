import streamlit as st
import helper
import Batsman_helper
import graphs
import Batsman_preprocessor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.sidebar.title("IPL Data Analyzer")

category = st.sidebar.radio(
    "Whom you want to analyse?",
    ('Baller', 'Batsman'))

if category == 'Baller':
    st.sidebar.write('You selected Baller.')
else:
    st.sidebar.write("You selected Batsman.")



option = st.sidebar.selectbox(
'How you want to analyse?',
('OVERALL', 'Yearly visualisation'))
st.sidebar.write('You selected:', option)

load = st.sidebar.button("Show Analysis")
load1 = st.sidebar.button("Make Win prediction")

#initialize session state
if "load_state" not in st.session_state:
    st.session_state.load_state = False

if load or st.session_state.load_state:
    st.session_state.load_state = True
    # baller analyses
    if category=='Baller' and option == 'OVERALL':
        Economy= helper.fetch_economy()
        Wicket_taking_ability=helper.fetch_ability()
        Consistency=helper.fetch_consistency()
        Critical_Wicket_Taking_Ability =helper.fetch_critical_ability()
        st.title("Baller Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Economy**")
            st.dataframe(Economy)
        with col2:
            st.markdown("**Wicket_taking_ability**")
            st.dataframe(Wicket_taking_ability)
        with col3:
            st.markdown("**Consistency**")
            st.dataframe(Consistency)
        with col4:
            st.markdown("**Critical_Wicket_Taker**")
            st.dataframe(Critical_Wicket_Taking_Ability)

        # Economy Variation
        st.title('Economy Variation')
        col1, col2 = st.columns(2)

        with col1:
            eco1=helper.fetch_eco1()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for bowler in eco1['bowler'].unique():
                x = eco1[eco1['bowler'] == bowler]['season']
                y = eco1[eco1['bowler'] == bowler]['economy']
                ax.plot(x, y, label=bowler)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('bowler economy')
            ax.set_title('bowler economy per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)


        with col2:
            eco2 = helper.fetch_eco2()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for bowler in eco2['bowler'].unique():
                x = eco2[eco2['bowler'] == bowler]['season']
                y = eco2[eco2['bowler'] == bowler]['economy']
                ax.plot(x, y, label=bowler)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('bowler economy')
            ax.set_title('bowler economy per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        ## yearly wickets taken
        st.title('Yearly wickets taken')
        col1, col2 = st.columns(2)

        with col1:
            wic1 = helper.fetch_wic1()


            plot=sns.catplot(x='bowler', y='player_dismissed', hue='season', kind='bar', data=wic1)

            # Set the plot title and axis labels
            plt.title('Number of wickets by bowler')
            plt.xlabel('Bowler')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            plt.show()

            # Display the graph
            st.pyplot(plot)

        with col2:
            wic2 = helper.fetch_wic2()
            # Create a figure and axis object
            plot = sns.catplot(x='bowler', y='player_dismissed', hue='season', kind='bar', data=wic2)

            # Set the plot title and axis labels
            plt.title('Number of wickets by bowler')
            plt.xlabel('Bowler')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            plt.show()

            # Display the graph
            st.pyplot(plot)

        ##yearly consistency
        st.title('Yearly consistency')
        col1, col2 = st.columns(2)

        with col1:
            con1 = helper.fetch_con1()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for bowler in con1['bowler'].unique():
                x = con1[con1['bowler'] == bowler]['season']
                y = con1[con1['bowler'] == bowler]['consistency']
                ax.plot(x, y, label=bowler)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('consistency')
            ax.set_title('consistency per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        with col2:
            con2 = helper.fetch_con2()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for bowler in con2['bowler'].unique():
                x = con2[con2['bowler'] == bowler]['season']
                y = con2[con2['bowler'] == bowler]['consistency']
                ax.plot(x, y, label=bowler)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('consistency')
            ax.set_title('consistency per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        ## Yearly num of times critical wickets taken
        st.title('Yearly num of times critical wickets taken')
        col1, col2 = st.columns(2)

        with col1:
            crt1 = helper.fetch_crt1()
            # Create a figure and axis object
            plot = sns.catplot(x='bowler', y='player_dismissed', hue='season', kind='bar', data=crt1)

            # Set the plot title and axis labels
            plt.title('Number of wickets by bowler')
            plt.xlabel('Bowler')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            plt.show()

            # Display the graph
            st.pyplot(plot)

        with col2:
            crt2 = helper.fetch_crt2()
            # Create a figure and axis object
            plot = sns.catplot(x='bowler', y='player_dismissed', hue='season', kind='bar', data=crt2)

            # Set the plot title and axis labels
            plt.title('Number of wickets by bowler')
            plt.xlabel('Bowler')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            plt.show()

            # Display the graph
            st.pyplot(plot)


    # Batsman Analyses
    elif category == 'Batsman' and option == 'OVERALL':
        deliveries = pd.read_csv('deliveries.csv')
        final = Batsman_preprocessor.preprocess(deliveries)
        hard_hitting_ability=Batsman_helper.fetch_hh_ability(final)
        finishing_ability=Batsman_helper.fetch_finishing_ability(final)
        consistency=Batsman_helper.fetch_consistency(final)
        running_runs=Batsman_helper.fetch_running_runs(final)
        st.title("Batsman Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Hard_hitting_ability**")
            st.dataframe(hard_hitting_ability)
        with col2:
            st.markdown("**Finishing_ability**")
            st.dataframe(finishing_ability)
        with col3:
            st.markdown("**Consistency**")
            st.dataframe(consistency)
        with col4:
            st.markdown("**running_runs**")
            st.dataframe(running_runs)

        #variation in runs of top batsman
        st.title('Batsman Runs variation')
        col1, col2= st.columns(2)

        with col1:
            yearly_top = graphs.fetch_top_scorer()
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in yearly_top['batsman'].unique():
                x = yearly_top[yearly_top['batsman'] == batsman]['season']
                y = yearly_top[yearly_top['batsman'] == batsman]['batsman_runs']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('Batsman Runs')
            ax.set_title('Batsman Runs per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        with col2:
            yearly_top2=graphs.fetch_top_scorer2()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in yearly_top2['batsman'].unique():
                x = yearly_top2[yearly_top2['batsman'] == batsman]['season']
                y = yearly_top2[yearly_top2['batsman'] == batsman]['batsman_runs']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('Batsman Runs')
            ax.set_title('Batsman Runs per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        # variation in strike rate of top batsman
        st.title('Batsman Strike rate variation')
        col1, col2 = st.columns(2)
        with col1:
            sr_top1=graphs.fetch_top_sr()
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in sr_top1['batsman'].unique():
                x = sr_top1[sr_top1['batsman'] == batsman]['season']
                y = sr_top1[sr_top1['batsman'] == batsman]['strike_rate']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('strike_rate')
            ax.set_title('Batsman strike rate per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        with col2:
            sr_top2 = graphs.fetch_top_sr2()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in sr_top2['batsman'].unique():
                x = sr_top2[sr_top2['batsman'] == batsman]['season']
                y = sr_top2[sr_top2['batsman'] == batsman]['strike_rate']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('strike_rate')
            ax.set_title('Batsman strike rate per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        # number of boundaries
        st.title('Number of boundaries')
        col1, col2 = st.columns(2)
        num1,num2 = Batsman_helper.fetch_num_boundries(final)
        with col1:
           # num1 = Batsman_helper.fetch_num_boundries(final)
            # Create a bar chart of the number of num_4_6 for the first 10 players
            fig, ax = plt.subplots()

            ax.bar(num1['batsman'], num1['num_4_6'])

            # Set the title and axis labels
            #plt.title('Number of num_4_6 for first 10 players')
            ax.set_xlabel('Batsman')
            ax.set_ylabel('Number of num_4_6')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=90)

            # Show the plot
            st.pyplot(fig)

        with col2:
           # num1 = Batsman_helper.fetch_num_boundries(final)
            # Create a bar chart of the number of num_4_6 for the first 10 players
            fig, ax = plt.subplots()

            ax.bar(num2['batsman'], num2['num_4_6'])

            # Set the title and axis labels
            # plt.title('Number of num_4_6 for first 10 players')
            ax.set_xlabel('Batsman')
            ax.set_ylabel('Number of num_4_6')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=90)

            # Show the plot
            st.pyplot(fig)

        # batting average
        st.title('Batting Average')
        col1, col2 = st.columns(2)
        with col1:
            avg_top1=graphs.fetch_top_batavg1()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in avg_top1['batsman'].unique():
                x = avg_top1[avg_top1['batsman'] == batsman]['season']
                y = avg_top1[avg_top1['batsman'] == batsman]['bat_avg']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('bat_avg')
            ax.set_title('Batting average per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        with col2:
            avg_top2 = graphs.fetch_top_batavg2()
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Iterate over each batsman, plot their runs by season, and label the line with their name
            for batsman in avg_top2['batsman'].unique():
                x = avg_top2[avg_top2['batsman'] == batsman]['season']
                y = avg_top2[avg_top2['batsman'] == batsman]['bat_avg']
                ax.plot(x, y, label=batsman)

            # Set the x-axis label, y-axis label, and title
            ax.set_xlabel('Season')
            ax.set_ylabel('bat_avg')
            ax.set_title('Batting average per Season in IPL')

            # Add a legend
            ax.legend()

            # Display the graph
            st.pyplot(fig)

        # Batting Strike Rate vs. Average:
        st.title('Batting Strike Rate vs Average')
        col1, col2 = st.columns(2)
        with col1:
                avg_top1 = graphs.fetch_top_batavg1()


                # Scatter plot with regression line of Batting Strike Rate vs. Average
                plot = sns.relplot(x='strike_rate', y='bat_avg', hue='batsman', data=avg_top1)
                # Set the title and axis labels
                plt.title('Batting Strike Rate vs. Average')
                plt.xlabel('Batting Strike Rate')
                plt.ylabel('Batting Average')

                # Show the plot
                st.pyplot(plot)

        with col2:
                avg_top2 = graphs.fetch_top_batavg2()

                # Scatter plot with regression line of Batting Strike Rate vs. Average
                plot = sns.relplot(x='strike_rate', y='bat_avg', hue='batsman', data=avg_top2)
                # Set the title and axis labels
                plt.title('Batting Strike Rate vs. Average')
                plt.xlabel('Batting Strike Rate')
                plt.ylabel('Batting Average')

                # Show the plot
                st.pyplot(plot)

        # Number of Sixes vs. Number of Fours
        st.title('Number of Sixes vs. Number of Fours')
        col1, col2 = st.columns(2)
        with col1:
            num_top1=graphs.fetch_top_four_sixes1()
            df_counts = num_top1[['batsman', 'num_4', 'num_6']]

            # Melt the dataframe to get the counts in a single column
            df_counts = df_counts.melt(id_vars=['batsman'], var_name='Boundary', value_name='Count')

            # Plot the count plot using seaborn
            plot=sns.catplot(x='batsman', y='Count', hue='Boundary', kind='bar', data=df_counts)

            # Set the plot title and axis labels
            plt.title('Number of 4s and 6s for each batsman')
            plt.xlabel('Boundary')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            st.pyplot(plot)

        with col2:
            num_top2=graphs.fetch_top_four_sixes2()
            df_counts = num_top2[['batsman', 'num_4', 'num_6']]

            # Melt the dataframe to get the counts in a single column
            df_counts = df_counts.melt(id_vars=['batsman'], var_name='Boundary', value_name='Count')

            # Plot the count plot using seaborn
            plot=sns.catplot(x='batsman', y='Count', hue='Boundary', kind='bar', data=df_counts)

            # Set the plot title and axis labels
            plt.title('Number of 4s and 6s for each batsman')
            plt.xlabel('Boundary')
            plt.ylabel('Count')

            # Rotate the x-axis labels for better visibility
            plt.xticks(rotation=45)

            # Show the plot
            st.pyplot(plot)

        # Batting Strike Rate vs. Runs Scored¶
        st.title('Batting Strike Rate vs. Runs Scored¶')
        col1, col2 = st.columns(2)
        with col1:

            sr_top1=graphs.fetch_top_sr()

            palette = sns.color_palette("muted", 5)
            fig = plt.figure()
            sns.scatterplot(x='strike_rate', y='batsman_runs', data=sr_top1, hue='batsman', style='season', palette=palette,
                            legend=True)
            # Set the title and axis labels

            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

            plt.title('Batting Strike Rate vs. batsman_runs')
            plt.xlabel('Batting Strike Rate')
            plt.ylabel('batsman_runs')


            st.pyplot(fig)
        with col2:

            sr_top2=graphs.fetch_top_sr2()
            sr_top1 = graphs.fetch_top_sr()

            palette = sns.color_palette("muted", 5)
            fig = plt.figure()
            sns.scatterplot(x='strike_rate', y='batsman_runs', data=sr_top2, hue='batsman', style='season',
                            palette=palette,
                            legend=True)
            # Set the title and axis labels

            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

            plt.title('Batting Strike Rate vs. batsman_runs')
            plt.xlabel('Batting Strike Rate')
            plt.ylabel('batsman_runs')

            st.pyplot(fig)

elif load1 or st.session_state.load_state:
    st.session_state.load_state = False

    teams = ['Sunrisers Hyderabad',
             'Mumbai Indians',
             'Royal Challengers Bangalore',
             'Kolkata Knight Riders',
             'Kings XI Punjab',
             'Chennai Super Kings',
             'Rajasthan Royals',
             'Delhi Capitals']

    cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
              'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
              'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
              'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
              'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
              'Sharjah', 'Mohali', 'Bengaluru']


    st.title('IPL Win Predictor')

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select the batting team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the bowling team', sorted(teams))

    selected_city = st.selectbox('Select host city', sorted(cities))

    target = st.number_input('Target')

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input('Score')
    with col4:
        overs = st.number_input('Overs completed')
    with col5:
        wickets = st.number_input('Wickets out')

    if st.button('Predict Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target],
             'crr': [crr], 'rrr': [rrr]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + "- " + str(round(win * 100)) + "%")
        st.header(bowling_team + "- " + str(round(loss * 100)) + "%")

























