"""Package of custom plotting functions"""

import itertools
import matplotlib as mpl
import matplotlib.pyplot as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import pandas as pd
import numpy as np
import numpy.ma as ma


__author__ = 'Michael N Rossol'


__all__ = ["COLORS", "get_COLORS", "def_linestyles", "def_markers", "riffle",
           "line_plot", "dual_plot", "error_plot", "contour_plot",
           "surface_plot", "colorbar"]


COLORS = {
    "red": (0.7176, 0.1098, 0.1098),
    "green": (0.65 * 0.298, 0.65 * 0.6863, 0.65 * 0.3137),
    "blue": (0.9 * 0.0824, 0.9 * 0.3961, 0.9 * 0.7529),
    "orange": (0.85 * 1.0, 0.85 * 0.5961, 0.0),
    "purple": (0.49412, 0.3412, 0.7608),
    "grey": (0.45, 0.45, 0.45),
    "cyan": (0.0, 0.7373, 0.8314),
    "teal": (0.0, 0.5882, 0.5333),
    "lime": (0.8039, 0.8627, 0.2235),
    "brown": (0.4745, 0.3333, 0.2824),
    "black": (0.0, 0.0, 0.0)
}


def get_COLORS(colors, n=None):
    """
    Parameters
    ----------
    colors : 'list'
        List of strings of color names
    n : 'int'
        repeat each color in colors n times
    Returns
    -------
    RGB color codes for plotting functions
    """
    if n is not None:
        colors = np.asarray([[color, ] * n for color in colors]).flatten()

    return [COLORS[color] for color in colors]


def_linestyles = ('-', '--', '-.', ':')
def_markers = (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h',
               u'H', u'D', u'd')


def riffle(*args):
    """
    Parameters
    ----------
    *args : 'Tuple'
        set of lists to be riffled together
    Returns
    -------
    Flattened list of lists such that entries are riffled
    """
    return [item for sublist in zip(*args) for item in sublist]


def line_plot(data, xlabel=None, ylabel=None, xlim=None, ylim=None,
              xticks=None, yticks=None, ticksize=(8, 2), colors=None,
              linestyles='Automatic', linewidth=2, markers=None, markersize=5,
              font='Arial', fontsize_axes=21, fontsize_other=18, borderwidth=2,
              add_legend=None, legend_location=0, figsize=(8, 6),
              resolution=300, showfig=True, filename=None):
    """
    Parameters
    ----------
    data : 'ndarray', shape(data[i]) = (n,2)
        Either a tuple or list of nx2 arrays or a single nx2 array.
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    colors : 'ndarray'
        Iterable list of colors to plot for each line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    linestyles : 'ndarray'
        Iterable list of Matplotlib designations for the linestyle for each
        line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    linewidth : 'Int'
        Line width for each line in 'data'.
    markersize : 'Float'
        Marker size for each marker in 'data'.
    markers : 'ndarray'
        Iterable list of Matplotlib designations for the marker for each line
        in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    boarderwidth : 'Int'
        Linewidth of plot frame
    add_legend : 'Bool', default = 'False'
        If 'True' a legend will be added at 'legendlocation'.
    legend_location : 'String' or 'Int', default = '0'
        Matplotlib designator for legend location, default is 'best'.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plots the lines in 'data'.
    """
    if not isinstance(data, (list, tuple)):
        lines = (data,)
    else:
        lines = data

    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = itertools.cycle((COLORS["blue"],
                                  COLORS["green"],
                                  COLORS["red"],
                                  COLORS["orange"],
                                  COLORS["purple"],
                                  COLORS["grey"],
                                  COLORS["cyan"],
                                  COLORS["teal"],
                                  COLORS["lime"],
                                  COLORS["brown"]))

    if linestyles is None:
        linestyles = itertools.cycle(('',))
    elif linestyles == 'Automatic':
        linestyles = itertools.cycle(def_linestyles)
    else:
        linestyles = itertools.cycle(linestyles)

    if markers is None:
        markers = itertools.cycle(('',))
    elif markers == 'Automatic':
        markers = itertools.cycle(def_markers)
    else:
        markers = itertools.cycle(markers)

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    for line in lines:
        axis.plot(line[:, 0], line[:, 1], linewidth=linewidth,
                  markersize=markersize, marker=next(markers),
                  color=next(colors), linestyle=next(linestyles))

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    # update plot labels and format based on user input
    for ax in ['top', 'bottom', 'left', 'right']:
        axis.spines[ax].set_linewidth(borderwidth)

    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize_axes)

    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize_axes)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.tick_params(axis='both', labelsize=fontsize_other, width=ticksize[1],
                     length=ticksize[0])

    if xticks is not None:
        axis.set_xticks(xticks)

    if yticks is not None:
        axis.set_yticks(yticks)

    if add_legend is not None:
        mplt.legend(add_legend, prop={'size': 12}, loc=legend_location)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def bar_chart(data, bar_gap=0.1, xlabel=None, ylabel=None, xlim=None,
              ylim=None, xticks=None, yticks=None, ticksize=(8, 2),
              colors=None, orientation='vertical', font='Arial',
              fontsize_axes=21, fontsize_other=18, borderwidth=2,
              add_legend=None, legend_location=0, figsize=(8, 6),
              resolution=300, showfig=True, filename=None):
    """
    Parameters
    ----------
    data : 'pandas Series or DataFrame',
        Either a pandas Series or DataFrame. Bars are values, xlabels are
        indeces.
    bar_gap : 'float'
        Spacing between bars as percentage of full width
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Default is the index of the Series
        or DataFrame.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    colors : 'ndarray'
        Iterable list of colors to plot for each line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    boarderwidth : 'Int'
        Linewidth of plot frame
    add_legend : 'Bool', default = 'False'
        If 'True' a legend will be added at 'legendlocation'.
    legend_location : 'String' or 'Int', default = '0'
        Matplotlib designator for legend location, default is 'best'.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plots a bar chart from the Series or DataFrame in 'data'.
    """

    if isinstance(data, pd.Series):
        bars = (data,)
    else:
        bars = [data[col].values for col in data.columns]

    xs = np.arange(len(data))

    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = itertools.cycle((COLORS["blue"],
                                  COLORS["green"],
                                  COLORS["red"],
                                  COLORS["orange"],
                                  COLORS["purple"],
                                  COLORS["grey"],
                                  COLORS["cyan"],
                                  COLORS["teal"],
                                  COLORS["lime"],
                                  COLORS["brown"]))

    bar_width = (1 - bar_gap*2)/len(bars)
    s_width = bar_gap
    c_shift = 0.5

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    for i, yheight in enumerate(bars):
        axis.bar(xs+s_width + bar_width*i, yheight, bar_width,
                 color=next(colors), orientation=orientation)

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    # update plot labels and format based on user input
    for ax in ['top', 'bottom', 'left', 'right']:
        axis.spines[ax].set_linewidth(borderwidth)

    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize_axes)

    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize_axes)

    if xlim is None:
        axis.set_xlim(xs[0], xs[-1] + 1)
    else:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.tick_params(axis='both', labelsize=fontsize_other, width=ticksize[1],
                     length=ticksize[0])

    axis.set_xticks(xs + c_shift)

    if xticks is None:
        axis.set_xticklabels(data.index, rotation=orientation)
    else:
        axis.set_xticklabels(xticks, rotation=orientation)

    if yticks is not None:
        axis.set_yticks(yticks)

    if add_legend is not None:
        mplt.legend(add_legend, prop={'size': 12}, loc=legend_location)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def dual_plot(data1, data2, xlabel=None, ylabel=None, xlim=None, ylim=None,
              xticks=None, yticks=None, ticksize=(8, 2), axis_colors='k',
              colors=None, linestyles='Automatic', linewidth=2, markersize=5,
              markers=None, font='Arial', fontsize_axes=21, fontsize_other=18,
              borderwidth=2, add_legend=None, legend_location=0,
              figsize=(8, 6), resolution=300, showfig=True, filename=None):
    """

    Parameters
    ----------
    data1 : 'ndarray', shape(data2[i]) = (n,2)
        Either a tuple or list of nx2 arrays or a single nx2 array.
    data2 : 'ndarray', shape(data2[i]) = (n,2)
        Either a tuple or list of nx2 arrays or a single nx2 array.
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    axis_colors : 'string', 'tuple'
        string indicating y-axis colors, tuple of strings indicates color of
        each y-axis
    colors : 'ndarray'
        Iterable list of colors to plot for each line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    linestyles : 'ndarray'
        Iterable list of Matplotlib designations for the linestyle for each
        line in 'data'. Will be cycled if fewer entries are specified than the
        number of lines in 'data'.
    linewidth : 'Int'
        Line width for each line in 'data'.
    markersize : 'Float'
        Marker size for each marker in 'data'.
    markers : 'ndarray'
        Iterable list of Matplotlib designations for the marker for each line
        in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    boarderwidth : 'Int'
        Linewidth of plot frame
    add_legend : 'Bool', default = 'False'
        If 'True' a legend will be added at 'legendlocation'.
    legend_location : 'String' or 'Int', default = '0'
        Matplotlib designator for legend location, default is 'best'.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plots the lines in 'data1' and lines in 'data2' on a dual y-axis plot.
    """
    if not isinstance(data1, (list, tuple)):
        lines1 = (data1,)
    else:
        lines1 = data1

    if not isinstance(data2, (list, tuple)):
        lines2 = (data2,)
    else:
        lines2 = data2

    if colors is not None:
        if isinstance(colors[0], list):
            colors1 = itertools.cycle(colors[0])
            colors2 = itertools.cycle(colors[1])
        else:
            colors1 = itertools.cycle(colors[::2])
            colors2 = itertools.cycle(colors[1::2])
    else:
        colors1 = itertools.cycle((COLORS["blue"],
                                  COLORS["red"],
                                  COLORS["purple"],
                                  COLORS["cyan"],
                                  COLORS["lime"]))

        colors2 = itertools.cycle((COLORS["green"],
                                  COLORS["orange"],
                                  COLORS["grey"],
                                  COLORS["teal"],
                                  COLORS["brown"]))

    if linestyles is None:
        linestyles1 = itertools.cycle(('',))
        linestyles2 = itertools.cycle(('',))
    elif linestyles == 'Automatic':
        linestyles1 = itertools.cycle(def_linestyles[::2])
        linestyles2 = itertools.cycle(def_linestyles[1::2])
    else:
        if isinstance(linestyles[0], (list, tuple)):
            linestyles1 = itertools.cycle(linestyles[0])
            linestyles2 = itertools.cycle(linestyles[1])
        else:
            linestyles1 = itertools.cycle(linestyles[::2])
            linestyles2 = itertools.cycle(linestyles[1::2])

    if markers is None:
        markers1 = itertools.cycle(('',))
        markers2 = itertools.cycle(('',))
    elif markers == 'Automatic':
        markers1 = itertools.cycle(def_markers[::2])
        markers2 = itertools.cycle(def_markers[1::2])
    else:
        if isinstance(markers[0], (list, tuple)):
            markers1 = itertools.cycle(markers[0])
            markers2 = itertools.cycle(markers[1])
        else:
            markers1 = itertools.cycle(markers[::2])
            markers2 = itertools.cycle(markers[1::2])

    if len(axis_colors) == 1:
        axis_color1 = axis_colors
        axis_color2 = axis_colors
    else:
        axis_color1 = axis_colors[0]
        axis_color2 = axis_colors[1]

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis1 = fig.add_subplot(111)

    for line in lines1:
        axis1.plot(line[:, 0], line[:, 1], linewidth=linewidth,
                   markersize=markersize, marker=next(markers1),
                   color=next(colors1), linestyle=next(linestyles1))

    axis2 = axis1.twinx()

    for line in lines2:
        axis2.plot(line[:, 0], line[:, 1], linewidth=linewidth,
                   markersize=markersize, marker=next(markers2),
                   color=next(colors2), linestyle=next(linestyles2))

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    # update plot labels and format based on user input
    for ax in ['top', 'bottom', 'left', 'right']:
        axis1.spines[ax].set_linewidth(borderwidth)
        axis2.spines[ax].set_linewidth(borderwidth)

    if xlabel is not None:
        axis1.set_xlabel(xlabel, fontsize=fontsize_axes)

    if ylabel is not None:
        if len(ylabel) == 1:
            axis1.set_ylabel(ylabel, fontsize=fontsize_axes, color=axis_color1)
            axis2.set_ylabel(ylabel, fontsize=fontsize_axes, color=axis_color2)
        else:
            axis1.set_ylabel(ylabel[0], fontsize=fontsize_axes,
                             color=axis_color1)
            axis2.set_ylabel(ylabel[1], fontsize=fontsize_axes,
                             color=axis_color2)

    if xlim is not None:
        axis1.set_xlim(xlim)

    if ylim is not None:
        if len(np.asarray(ylim).shape) == 1:
            axis1.set_ylim(ylim)
            axis2.set_ylim(ylim)
        else:
            axis1.set_ylim(ylim[0])
            axis2.set_ylim(ylim[1])

    if xticks is not None:
        axis1.set_xticks(xticks)

    if yticks is not None:
        if len(np.asarray(yticks).shape) == 1:
            axis1.set_set_yticks(yticks)
            axis2.set_set_yticks(yticks)
        else:
            axis1.set_set_yticks(yticks[0])
            axis2.set_set_yticks(yticks[1])

    axis1.tick_params(axis='x', labelsize=fontsize_other, width=ticksize[1],
                      length=ticksize[0], color='k')
    axis1.tick_params(axis='y', labelsize=fontsize_other, width=ticksize[1],
                      length=ticksize[0], color=axis_color1)
    axis2.tick_params(axis='y', labelsize=fontsize_other, width=ticksize[1],
                      length=ticksize[0], color=axis_color2)

    if len(axis_colors) == 2:
        for tl in axis1.get_yticklabels():
            tl.set_color(axis_colors[0])
        for t2 in axis2.get_yticklabels():
            t2.set_color(axis_colors[1])

    if add_legend is not None:
        mplt.legend(add_legend, prop={'size': 12}, loc=legend_location)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def error_plot(data, xlabel=None, ylabel=None, xlim=None, ylim=None,
               xticks=None, yticks=None, ticksize=(8, 2), colors=None,
               linestyles=None, linewidth=2, markersize=5,
               markers='Automatic', font='Arial', fontsize_axes=21,
               fontsize_other=18, borderwidth=2, add_legend=None,
               legend_location=0, figsize=(8, 6), resolution=300,
               showfig=True, filename=None):
    """
    Parameters
    ----------
    data : 'ndarray', shape(data[i]) = (n,2,2)
        Either a tuple or list of nx2X2 arrays or a single nx2X2 array.
        data given as an array or list of [[x, sigma_x], [y, sigma_y]
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    colors : 'ndarray'
        Iterable list of colors to plot for each line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    linestyles : 'ndarray'
        Iterable list of Matplotlib designations for the linestyle for each
        line in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    linewidth : 'Int'
        Line width for each line in 'data'.
    markersize : 'Float'
        Marker size for each marker in 'data'.
    markers : 'ndarray'
        Iterable list of Matplotlib designations for the marker for each line
        in 'data'.
        Will be cycled if fewer entries are specified than the number of lines
        in 'data'.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    boarderwidth : 'Int'
        Linewidth of plot frame
    add_legend : 'Bool', default = 'False'
        If 'True' a legend will be added at 'legendlocation'.
    legend_location : 'String' or 'Int', default = '0'
        Matplotlib designator for legend location, default is 'best'.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plots the lines in 'data' with given error bars.
    """
    if not isinstance(data, (list, tuple)):
        lines = (data,)
    else:
        lines = data

    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = itertools.cycle((COLORS["blue"],
                                  COLORS["green"],
                                  COLORS["red"],
                                  COLORS["orange"],
                                  COLORS["purple"],
                                  COLORS["grey"],
                                  COLORS["cyan"],
                                  COLORS["teal"],
                                  COLORS["lime"],
                                  COLORS["brown"]))

    if linestyles is None:
        linestyles = itertools.cycle(('',))
    elif linestyles == 'Automatic':
        linestyles = itertools.cycle(def_linestyles)
    else:
        linestyles = itertools.cycle(linestyles)

    if markers is None:
        markers = itertools.cycle(('',))
    elif markers == 'Automatic':
        markers = itertools.cycle(def_markers)
    else:
        markers = itertools.cycle(markers)

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    for line in lines:
        x = line[:, 0, 0]
        error = line[:, 0, 1]
        if np.isnan(error).all():
            x_error = None
        else:
            x_error = error

        y = line[:, 1, 0]
        error = line[:, 1, 1]
        if np.isnan(error).all():
            y_error = None
        else:
            y_error = error

        axis.errorbar(x, y, xerr=x_error, yerr=y_error, linewidth=linewidth,
                      markersize=markersize, marker=next(markers),
                      color=next(colors), linestyle=next(linestyles))

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    # update plot labels and format based on user input
    for ax in ['top', 'bottom', 'left', 'right']:
        axis.spines[ax].set_linewidth(borderwidth)

    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize_axes)

    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize_axes)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.tick_params(axis='both', labelsize=fontsize_other, width=ticksize[1],
                     length=ticksize[0])

    if xticks is not None:
        axis.set_xticks(xticks)

    if yticks is not None:
        axis.set_yticks(yticks)

    if add_legend is not None:
        mplt.legend(add_legend, prop={'size': 12}, loc=legend_location)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def contour_plot(data, xlim=None, ylim=None, zlim=None, major_spacing=None,
                 minor_spacing=None, contour_width=1, contour_color='k',
                 opacity=1., colorbar_on=True, colorbar_location='right',
                 colorbar_label=None, colorbar_lines=True,
                 colorbar_ticks=None, colormap=None, font='Arial',
                 fontsize_axes=21, fontsize_other=18, fontsize_colorbar=21,
                 axis_on=False, xlabel=None, ylabel=None, xticks=None,
                 yticks=None, ticksize=(8, 2), borderwidth=2, figsize=6,
                 resolution=300, showfig=True, filename=None):
    """
    Parameters
    ----------
    data : 'ndarray', len(data) = 3, shape(data[i]) = (n,m)
        Either a tuple or list of nx2 arrays or a single nx2 array.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    zlim : 'tuple', len(ylim) == 2
        Upper and lower limits for the z-data.
    major_spacing : 'float'
        Spacing between major contours.
    minor_spacing :  'float'
        Spacing between minor contours.
    contour_width : 'int'
        Width of contour lines
    contour_color : 'string'
        Color of contour lines
    opacity : 'float'
        Opacity of contour plot (0 = transparent)
    colorbar_on : 'boole', default=True
        Show colorbar.
    colorbar_location : 'string'
        Location of colorbar with respect to contour_plot
    colorbar_label : 'string'
        Label for colorbar.
    colorbar_lines : 'boole'
        Show contour lines on colorbar
    colormap : 'string'
        Colormap name
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    fontsize_colorbar : 'Int'
        Font size to be used for colorbar label.
    axis_on : 'boole', default=false
        Show axis.
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    boarderwidth : 'Int'
        Linewidth of plot frame
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plot (x, y, z) arrays in 'data' as filled contours.
    """

    assert len(data) == 3, 'Data must equal (x, y, data)'

    x, y, z = data
    z_m = ma.masked_invalid(z)

    a_ratio = z.shape
    a_ratio = a_ratio[1]/a_ratio[0]

    if isinstance(figsize, (int, float)):
        figsize = [figsize*a_ratio, figsize]
    else:
        figsize = max(figsize)
        figsize = [figsize*a_ratio, figsize]

    if colorbar_on:
        if colorbar_location in ['top', 'bottom']:
            figsize[1] += figsize[1]/10
            cbar_size = figsize[0]/20
        else:
            figsize[0] += figsize[0]/10
            cbar_size = figsize[1]/20
    else:
        cbar_size = max(figsize)/20

    if zlim is None:
        zmin, zmax = np.nanmin(z), np.nanmax(z)
    else:
        zmin, zmax = zlim

    if major_spacing is None:
        major_spacing = (zmax - zmin)/10
    if minor_spacing is None:
        minor_spacing = major_spacing/10

    cl_levels = np.arange(zmin, zmax + major_spacing, major_spacing)
    cf_levels = np.arange(zmin, zmax + minor_spacing, minor_spacing)

    if colorbar_ticks is None:
        l_levels = cl_levels[::2]
    else:
        l_levels = (zmax - zmin)/colorbar_ticks
        l_levels = np.arange(zmin, zmax + l_levels, l_levels)

    orientation = 'vertical'
    if colorbar_location in ['top', 'bottom']:
        orientation = 'horizontal'

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    cf = mplt.contourf(x, y, z_m, alpha=opacity, levels=cf_levels,
                       extend='both', antialiased=True)

    if contour_color is not None:
        cl = mplt.contour(cf, levels=cl_levels, colors=(contour_color,),
                          linewidths=(contour_width,))

    if colormap is not None:
        cf.set_cmap(colormap)

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42
    mplt.axes().set_aspect('equal')

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    cbar_padding = 0.1

    if axis_on:
        for ax in ['top', 'bottom', 'left', 'right']:
            axis.spines[ax].set_linewidth(borderwidth)

        if xlabel is not None:
            axis.set_xlabel(xlabel, fontsize=fontsize_axes)

        if ylabel is not None:
            axis.set_ylabel(ylabel, fontsize=fontsize_axes)

        axis.tick_params(axis='both', labelsize=fontsize_other,
                         width=ticksize[1], length=ticksize[0])

        if xticks is not None:
            axis.set_xticks(xticks)

        if yticks is not None:
            axis.set_yticks(yticks)

        if colorbar_location == 'bottom':
            if xlabel is None:
                cbar_padding = 0.4
            else:
                cbar_padding = 0.8
        elif colorbar_location == 'left':
            if ylabel is None:
                cbar_padding = 0.7
            else:
                cbar_padding = 1.1

    else:
        mplt.axis('off')

    if colorbar_on:
        divider = make_axes_locatable(axis)

        caxis = divider.append_axes(colorbar_location, size=cbar_size,
                                    pad=cbar_padding)

        cbar = mplt.colorbar(cf, ticks=l_levels, cax=caxis,
                             orientation=orientation,
                             ticklocation=colorbar_location)

        cbar.ax.tick_params(labelsize=fontsize_other)

        if colorbar_label is not None:
            cbar.set_label(colorbar_label, size=fontsize_colorbar)

        if colorbar_lines is not None:
            if contour_color is not None:
                cbar.add_lines(cl)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def scatter_plot(data, xlim=None, ylim=None, zlim=None, colorbar_on=True,
                 colorbar_location='right', colorbar_label=None,
                 colorbar_lines=True, colorbar_ticks=None, colormap=None,
                 font='Arial', fontsize_axes=21, fontsize_other=18,
                 fontsize_colorbar=21, xlabel=None, ylabel=None, xticks=None,
                 yticks=None, ticksize=(8, 2), borderwidth=2, figsize=(8, 6),
                 resolution=300, showfig=True, filename=None):
    """
    Parameters
    ----------
    data : 'ndarray'
        nx3 array of data
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    zlim : 'tuple', len(ylim) == 2
        Upper and lower limits for the colorbar.
    colorbar_on : 'boole', default=True
        Show colorbar.
    colorbar_location : 'string'
        Location of colorbar with respect to contour_plot
    colorbar_label : 'string'
        Label for colorbar
    colormap : 'string'
        Colormap name
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    fontsize_colorbar : 'Int'
        Font size to be used for colorbar label.
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    boarderwidth : 'Int'
        Linewidth of plot frame
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plot (x, y, z) arrays in 'data' as filled contours.
    """

    assert data.shape[-1] == 3, 'Data must equal (x, y, data)'

    x = data[:, 0]
    y = data[:, 1]
    c = data[:, 2]
    zmax, zmin = c.max(), c.min()

    if isinstance(figsize, tuple):
        figsize = list(figsize)

    if colorbar_on:
        if colorbar_location in ['top', 'bottom']:
            figsize[1] += figsize[1]/10
            cbar_size = figsize[0]/20
        else:
            figsize[0] += figsize[0]/10
            cbar_size = figsize[1]/20

        if zlim is None:
            zmax, zmin = c.max(), c.min()
        else:
            zmax, zmin = zlim

        if colorbar_ticks is None:
            l_levels = None
        else:

            l_levels = (zmax - zmin)/colorbar_ticks
            l_levels = np.arange(zmin, zmax + l_levels, l_levels)
    else:
        cbar_size = max(figsize)/20

    orientation = 'vertical'
    if colorbar_location in ['top', 'bottom']:
        orientation = 'horizontal'

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    s = mplt.scatter(x, y, c=c, norm=mpl.colors.Normalize(vmin=zmin,
                                                          vmax=zmax))

    if colormap is not None:
        s.set_cmap(colormap)

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    for ax in ['top', 'bottom', 'left', 'right']:
        axis.spines[ax].set_linewidth(borderwidth)

    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize_axes)

    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize_axes)

    axis.tick_params(axis='both', labelsize=fontsize_other,
                     width=ticksize[1], length=ticksize[0])

    if xticks is not None:
        axis.set_xticks(xticks)

    if yticks is not None:
        axis.set_yticks(yticks)

    if colorbar_on:
        cbar_padding = 0.1
        if colorbar_location == 'bottom':
            if xlabel is None:
                cbar_padding = 0.4
            else:
                cbar_padding = 0.8
        elif colorbar_location == 'left':
            if ylabel is None:
                cbar_padding = 0.7
            else:
                cbar_padding = 1.1
        divider = make_axes_locatable(axis)

        caxis = divider.append_axes(colorbar_location, size=cbar_size,
                                    pad=cbar_padding)

        cbar = mplt.colorbar(s, ticks=l_levels, cax=caxis,
                             orientation=orientation,
                             ticklocation=colorbar_location)

        cbar.ax.tick_params(labelsize=fontsize_other)

        if colorbar_label is not None:
            cbar.set_label(colorbar_label, size=fontsize_colorbar)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def surface_plot(data, xlim=None, ylim=None, zlim=None, stride=1,
                 box_ratio='Automatic', colorbar_on=True, colorbar_label=None,
                 colorbar_ticks=None, colormap=None, font='Arial',
                 fontsize_axes=21, fontsize_other=18, fontsize_colorbar=21,
                 axis_on=False, xlabel=None, ylabel=None, zlabel=None,
                 xticks=None, yticks=None, zticks=None, ticksize=(8, 2),
                 borderwidth=2, figsize=6, resolution=300, showfig=True,
                 filename=None):
    """
    Parameters
    ----------
    data : 'ndarray', len(data) = 3, shape(data[i]) = (n,m)
        Either a tuple or list of nx2 arrays or a single nx2 array.
    xlim : 'tuple', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'tuple', len(ylim) == 2
        Upper and lower limits for the y-axis.
    zlim : 'tuple', len(ylim) == 2
        Upper and lower limits for the z-data.
    major_spacing : 'float'
        Spacing between major contours.
    minor_spacing :  'float'
        Spacing between minor contours.
    contour_width : 'int'
        Width of contour lines
    contour_color : 'string'
        Color of contour lines
    opacity : 'float'
        Opacity of contour plot (0 = transparent)
    colorbar_on : 'boole', default=True
        Show colorbar.
    colorbar_location : 'string'
        Location of colorbar with respect to contour_plot
    colorbar_label : 'string'
        Label for colorbar.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    fontsize_colorbar : 'Int'
        Font size to be used for colorbar label.
    axis_on : 'boole', default=false
        Show axis.
    xlabel : 'String'
        Label for x-axis.
    ylabel : 'String'
        Label for y-axis.
    xticks : 'ndarray'
        List of ticks to use on the x-axis. Should be within the bounds set by
        xlim.
    yticks : 'ndarray'
        List of ticks to use on the y-axis. Should be within the bound set by
        ylim.
    ticksize : 'ndarray', default '[8,2]'
        Length and width of ticks.
    boarderwidth : 'Int'
        Linewidth of plot frame
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plot (x, y, z) arrays in 'data' as a three dimensional surface plot.
    """

    assert len(data) == 3, 'Data must equal (x, y, data)'

    x, y, z = data
    z_m = ma.masked_invalid(z)

    if box_ratio is None:
        data_ratio = [1, 1, 1]
    elif isinstance(box_ratio, list):
        data_ratio = box_ratio
    else:
        bad_pos = np.isnan(z)

        if xlim is None:
            x_range = copy.copy(x)
            x_range[bad_pos] = np.nan
            x_range = np.nanmax(x_range) - np.nanmin(x_range)
        else:
            x_range = xlim[1] - xlim[0]

        if ylim is None:
            y_range = copy.copy(y)
            y_range[bad_pos] = np.nan
            y_range = np.nanmax(y_range) - np.nanmin(y_range)
        else:
            y_range = ylim[1] - ylim[0]

        if zlim is None:
            z_range = np.nanmax(z) - np.nanmin(z)
        else:
            z_range = zlim[1] - zlim[0]

        data_ratio = [range/x_range for range in [x_range, y_range, z_range]]

    if isinstance(figsize, (list, tuple)):
        figsize = max(figsize)

    figsize = [figsize, figsize]

    if colorbar_on:
        figsize[0] += figsize[0]/10

    if zlim is None:
        zmin, zmax = np.nanmin(z), np.nanmax(z)
    else:
        zmin, zmax = zlim

    if colorbar_ticks is None:
        colorbar_ticks = 10

    l_levels = (zmax - zmin)/colorbar_ticks
    l_levels = np.arange(zmin, zmax + l_levels, l_levels)

    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax, clip=True)

    if colormap is None:
        cmap = mplt.cm.jet
    else:
        cmap = colormap

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.gca(projection='3d')

    surf = axis.plot_surface(x, y, z_m, rstride=stride, cstride=stride,
                             linewidth=0, antialiased=False, cmap=cmap,
                             norm=norm)

    axis.pbaspect = data_ratio

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    if zlim is not None:
        axis.set_zlim(zlim)

    if axis_on:
        for ax in ['top', 'bottom', 'left', 'right']:
            axis.spines[ax].set_linewidth(borderwidth)

        if xlabel is not None:
            axis.set_xlabel(xlabel, fontsize=fontsize_axes)

        if ylabel is not None:
            axis.set_ylabel(ylabel, fontsize=fontsize_axes)

        if zlabel is not None:
            axis.set_zlabel(zlabel, fontsize=fontsize_axes)

        axis.tick_params(axis='both', labelsize=fontsize_other,
                         width=ticksize[1], length=ticksize[0])

        if xticks is not None:
            axis.set_xticks(xticks)

        if yticks is not None:
            axis.set_yticks(yticks)

        if zticks is not None:
            axis.set_zticks(zticks)

    else:
        axis.axis('off')

    if colorbar_on:
        cbar = mplt.colorbar(surf, ticks=l_levels, orientation='vertical',
                             ticklocation='right', shrink=1)

        cbar.ax.tick_params(labelsize=fontsize_other)

        if colorbar_label is not None:
            cbar.set_label(colorbar_label, size=fontsize_colorbar)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()


def colorbar(zlim, ticks=None, lines=None, line_color='k', linewidth=1,
             colormap=None, extend='neither', ticklocation='right',
             fontsize_other=18, label=None, fontsize_label=21, figsize=6,
             resolution=300, showfig=True, filename=None):

    """
    Parameters
    ----------
    zlim : 'tuple'
        List or tuple indicating zmin and zmax.
    tick : 'Int'
        Number of ticks to label.
    lines : 'Int'
        Number of lines to draw on colorbar.
    line_color : 'String'
        Color of lines drawn on colorbar.
    linewidth : 'Int'
        Line width for each line drawn on colorbar.
    colormap : 'String'
        Color scheme for colorbar.
    extend : 'String'
        Direction to extend colors beyond zmin and zmax.
    ticklocation : 'String'
        Orientation of colorbar and location of tick marks.
    fontsize_other : 'Int'
        Font size of tick numbers.
    label : 'String'
        Label for colorbar
    fontsize_label : 'Int'
        Font size of label.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.

    Returns
    -------
    Plot custom colorbar
    """

    a_ratio = 20

    if isinstance(figsize, (list, tuple)):
        figsize = max(figsize)

    if ticklocation in ['right', 'left']:
        figsize = (figsize/a_ratio, figsize)
        orientation = 'vertical'
    else:
        figsize = (figsize, figsize/a_ratio)
        orientation = 'horizontal'

    if ticks is not None:
        ticks = (zlim[1] - zlim[0])/ticks
        ticks = np.arange(zlim[0], zlim[1] + ticks, ticks)

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    norm = mpl.colors.Normalize(vmin=zlim[0], vmax=zlim[1])

    cb = mpl.colorbar.ColorbarBase(axis, cmap=colormap, norm=norm,
                                   orientation=orientation, extend=extend,
                                   ticks=ticks, ticklocation=ticklocation)
    cb.ax.tick_params(labelsize=fontsize_other)

    if label is not None:
        cb.set_label(label, size=fontsize_label)

    if lines is not None:
        lines = (zlim[1] - zlim[0])/lines
        lines = np.arange(zlim[0], zlim[1] + lines, lines)
        cb.add_lines(lines, colors=(line_color,)*len(lines),
                     linewidths=(linewidth,)*len(lines))

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True,
                     bbox_inches='tight')

    if showfig:
        mplt.show()
