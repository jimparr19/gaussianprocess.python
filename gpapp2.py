import numpy as np

from bokeh.models import ColumnDataSource, TapTool, Circle
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import Slider
from bokeh.layouts import row, widgetbox

min_x = -3
max_x = 3
min_y = -5
max_y = 5


length_scale = Slider(title="length_scale", value=1.0, start=0.01, end=2.0, step=0.01)
power = Slider(title="power", value=2.0, start=1.0, end=3.0, step=0.1)
regression = Slider(title="regression", value=0.0, start=0.0, end=0.1, step=0.001)

l = length_scale.value
p_l = power.value
r = regression.value

xi = np.linspace(min_x, max_x, 100)
yi = np.linspace(min_y, max_y, 100)
xv, yv = np.meshgrid(xi, yi)
x_grid = xv.flatten()
y_grid = yv.flatten()

n = 50
x = np.linspace(min_x, max_x, n)
mu = np.zeros(n)

sigma_test = np.zeros((n,n))
for i in range(len(x)):
    for j in range(len(x)):
        sigma_test[i,j] = np.exp(-0.5 * (np.abs(x[i] - x[j])/l) ** p_l)
#sigma_test = sigma_test + r*np.eye(n)

err = sigma_test.diagonal()

upper_limit = mu + 1.95*err
lower_limit = mu - 1.95*err

band_x = np.append(x, x[::-1]).tolist()
band_y = np.append(lower_limit, upper_limit[::-1]).tolist()

TOOLS="reset"

# create the scatter plot
p = figure(tools=TOOLS, plot_width=800, plot_height=600, min_border=10, min_border_left=50,
           toolbar_location="above", title="GP updates")

p.background_fill_color = "#fafafa"


s1 = ColumnDataSource(data = dict(x=x_grid, y=y_grid))
s2 = ColumnDataSource(data = dict(x=[], y=[]))
s3 = ColumnDataSource(data = dict(x=x, y=mu, band_x=band_x, band_y=band_y))

r1 = p.circle(x='x', y='y', source=s1, line_color=None, fill_color=None, size=10, name="mycircle")
selected_circle = Circle(line_color=None, fill_color=None)
nonselected_circle = Circle(line_color=None, fill_color=None)
renderer = p.select(name="mycircle")
renderer.selection_glyph = selected_circle
renderer.nonselection_glyph = nonselected_circle
tap_tool = TapTool(renderers=[r1])
p.tools.append(tap_tool)

r2 = p.circle(x='x',y='y', source=s2, line_color='#d3d3d3', fill_color='#d3d3d3', size=10)
r3 = p.line(x='x',y='y', source=s3)

p.patch(x='band_x', y='band_y', source=s3, color='#d3d3d3', alpha=0.5)

ds = r2.data_source
gp = r3.data_source

def update_data(attr, old, new):

    l = length_scale.value
    p_l = power.value
    r = regression.value

    sigma_test = np.zeros((n, n))
    for i in range(len(x)):
        for j in range(len(x)):
            sigma_test[i, j] = np.exp(-0.5 * (np.abs(x[i] - x[j])/l) ** p_l)
    #sigma_test = sigma_test + r * np.eye(n)

    inds = new['1d']['indices']
    if inds != []:
        new_data = dict()
        new_data['x'] = ds.data['x'] + [x_grid[inds[0]]]
        new_data['y'] = ds.data['y'] + [y_grid[inds[0]]]
        ds.data = new_data

        x_obs = np.array(new_data['x'])
        y_obs = np.array(new_data['y'])

        sigma_train = np.zeros((len(x_obs), len(x_obs)))
        for i in range(len(x_obs)):
            for j in range(len(x_obs)):
                sigma_train[i, j] = np.exp(-0.5 * (np.abs(x_obs[i] - x_obs[j])/l) ** p_l)
        sigma_train = sigma_train + r * np.eye(len(x_obs))

        sigma_train_test = np.zeros((len(x_obs), len(x)))
        for i in range(len(x_obs)):
            for j in range(len(x)):
                sigma_train_test[i, j] = np.exp(-0.5 * (np.abs(x_obs[i] - x[j])/l) ** p_l)

        sigma_test_train = np.zeros((len(x), len(x_obs)))
        for i in range(len(x)):
            for j in range(len(x_obs)):
                sigma_test_train[i, j] = np.exp(-0.5 * (np.abs(x[i] - x_obs[j])/l) ** p_l)

        phi = sigma_test_train.dot(np.linalg.inv(sigma_train))

        pred = phi.dot(y_obs)
        cov = sigma_test - phi.dot(sigma_train_test)
        err = cov.diagonal()

        upper_limit = pred + 1.95*err
        lower_limit = pred - 1.95*err

        band_x = np.append(x, x[::-1]).tolist()
        band_y = np.append(lower_limit, upper_limit[::-1]).tolist()

        gp_data = dict()
        gp_data['x'] = x.tolist()
        gp_data['y'] = pred.tolist()
        gp_data['band_x'] = band_x
        gp_data['band_y'] = band_y
        gp.data = gp_data


def update_slider(attr, old, new):

    l = length_scale.value
    p_l = power.value
    r = regression.value

    sigma_test = np.zeros((n, n))
    for i in range(len(x)):
        for j in range(len(x)):
            sigma_test[i, j] = np.exp(-0.5 * (np.abs(x[i] - x[j])/l) ** p_l)
    #sigma_test = sigma_test + r * np.eye(n)

    x_obs = ds.data['x']
    y_obs = ds.data['y']

    sigma_train = np.zeros((len(x_obs), len(x_obs)))
    for i in range(len(x_obs)):
        for j in range(len(x_obs)):
            sigma_train[i, j] = np.exp(-0.5 * (np.abs(x_obs[i] - x_obs[j])/l) ** p_l)
    sigma_train = sigma_train + r * np.eye(len(x_obs))

    sigma_train_test = np.zeros((len(x_obs), len(x)))
    for i in range(len(x_obs)):
        for j in range(len(x)):
            sigma_train_test[i, j] = np.exp(-0.5 * (np.abs(x_obs[i] - x[j])/l) ** p_l)

    sigma_test_train = np.zeros((len(x), len(x_obs)))
    for i in range(len(x)):
        for j in range(len(x_obs)):
            sigma_test_train[i, j] = np.exp(-0.5 * (np.abs(x[i] - x_obs[j])/l) ** p_l)

    phi = sigma_test_train.dot(np.linalg.inv(sigma_train))

    pred = phi.dot(y_obs)
    cov = sigma_test - phi.dot(sigma_train_test)
    err = cov.diagonal()

    upper_limit = pred + 1.95*err
    lower_limit = pred - 1.95*err

    band_x = np.append(x, x[::-1]).tolist()
    band_y = np.append(lower_limit, upper_limit[::-1]).tolist()

    gp_data = dict()
    gp_data['x'] = x.tolist()
    gp_data['y'] = pred.tolist()
    gp_data['band_x'] = band_x
    gp_data['band_y'] = band_y
    gp.data = gp_data


r1.data_source.on_change('selected', update_data)

for w in [length_scale, power, regression]:
    w.on_change('value', update_slider)


inputs = widgetbox(length_scale, power, regression)

curdoc().add_root(row(inputs, p, width=1200))
curdoc().title = "Gaussian Process Updates"