# helper code, would need to be copied over into kickstarter_predictor.py to work

# plot % of success campaigns based on positivity score
main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(50,200,50)'})
# Replacing unknown value to nan
dataset['bool_flesch'] = dataset['bool_flesch'].replace('N,0"', np.nan)

data = []
total_expected_values = []
annotations = []
shapes = []

rate_success_positivity = dataset[dataset['state'] == 1].groupby(['bool_flesch']).count()['ID']\
                / dataset.groupby(['bool_flesch']).count()['ID'] * 100
rate_failed_positivity = dataset[dataset['state'] == 0].groupby(['bool_flesch']).count()['ID']\
                / dataset.groupby(['bool_flesch']).count()['ID'] * 100
    
rate_success_positivity = rate_success_positivity.sort_values(ascending=False)
rate_failed_positivity = rate_failed_positivity.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_positivity.index,
        y=rate_success_positivity,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_positivity.index,
        y=rate_failed_positivity,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by positivity',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations,
    shapes=shapes
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='main_pos')
