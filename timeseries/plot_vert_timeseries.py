import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

year = 2015
targets = [811, 815, 823, 911]
observatories = ["Fukuoka", "Ohashi"]
analyses = ["FNL", "ERA", "GPV"]
variables = {"WindSpeed": "Wind Speed [m/s]", "WindDirection": "Wind Direction [°]", "u": "$\it{u}}$ [m/s]", "v": "$\it{v}}$ [m/s]", "PotentialTemperature": "Potential Temperature [K]", "MixingRatio": "Mixing Ratio [kg/kg]"}

graph_format = ".pdf"  # ".png", ".svg", ".pdf", etc.

plt.rcParams["font.family"] = "Arial"
if graph_format == ".svg":
    plt.rcParams["svg.fonttype"] = "none"
if graph_format == ".pdf":
    plt.rcParams["pdf.fonttype"] = 42

figsize_setting = (15, 5)
left_adjust = 0.05
right_adjust = 0.80
bottom_adjust = 0.11
top_adjust = 0.95


def calc_u(speed, direction):
    return - speed * np.sin(np.deg2rad(direction))

def calc_v(speed, direction):
    return - speed * np.cos(np.deg2rad(direction))

def set_axes(variable, ymin, ymax, iplot):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.grid(zorder=0)

    if variable == "WindDirection":
        ax.set_ylim([-10, 370])
        ax.set_yticks(np.linspace(0.0, 360.0, 9))
        ax.set_yticks(np.linspace(22.5, 337.5, 8), minor=True)
        ax.grid(which="minor", axis="y", color="lightgray", linestyle="--", linewidth=1)
        ax2 = ax.twinx()
        ax2.set_ylim([-10, 370])
        ax2.set_yticks(np.linspace(0.0, 360.0, 17))
        ax2.set_yticklabels(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"], ha="right")
        ax2.tick_params(direction="in", length=0.0, pad=-2.0)
    elif variable == "WindSpeed":
        if iplot == 0:
            ymin, ymax = ax.get_ylim()
            ymin = 0.0
            ymax = int(ymax) + 1
        ax.set_ylim([ymin, ymax])
    elif variable == "MixingRatio":
        if iplot == 0:
            ymin, ymax = ax.get_ylim()
            ymin = int(ymin * 1000.0) * 0.001
            ymax = (int(ymax * 1000.0) + 1) * 0.001
        ax.set_ylim([ymin, ymax])
    else:
        if iplot == 0:
            ymin, ymax = ax.get_ylim()
            if ymin < 0:
                ymin = int(ymin) - 1
                ax.axhline(y=0, color="black", linewidth=1.0, zorder=10)
            else:
                ymin = int(ymin)
            ymax = int(ymax) + 1
        else:
            if ymin < 0:
                ax.axhline(y=0, color="black", linewidth=1.0, zorder=10)
        ax.set_ylim([ymin, ymax])

    return ymin, ymax, iplot + 1


for target in targets:
    if target == 911:
        spinups = [1, 2, 3, 5, 7, 10, 20, 30, 60]
    else:
        spinups = [1, 2, 3, 5, 7, 10, 20, 30]
    date_str = str(year) + "-" + str(target).zfill(4)[:2] + "-" + str(target).zfill(4)[2:]

    for observatory in observatories:
        work_dir = date_str + "/" + observatory + "/vert/"
        os.makedirs(work_dir + "timeseries_plots", exist_ok=True)
        graph_dir = work_dir + "timeseries_plots/"

        for k in range(12):
            height = []
            for analysis in analyses:
                for spinup in spinups:
                    csv_file = work_dir + "data/" + date_str + "_" + observatory + "_" + analysis + "_" + str(spinup).zfill(2) + "d15h_vert_TimeSeries_10min.csv"
                    csv_data = pd.read_csv(csv_file, header = 0, index_col = [0], parse_dates = [0])
                    height.append(csv_data.loc[:, "Height_" + str(k + 1)].mean())
                    if analyses.index(analysis) == 0 and spinups.index(spinup) == 0:
                        wsp_data = pd.DataFrame(csv_data.loc[:, "WindSpeed_" + str(k + 1)])
                        wdr_data = pd.DataFrame(csv_data.loc[:, "WindDirection_" + str(k + 1)])
                        u_data = pd.DataFrame(index=wsp_data.index, columns=[])
                        v_data = pd.DataFrame(index=wsp_data.index, columns=[])
                        th_data = pd.DataFrame(csv_data.loc[:, "PotentialTemperature_" + str(k + 1)])
                        qv_data = pd.DataFrame(csv_data.loc[:, "MixingRatio_" + str(k + 1)])
                    else:
                        wsp_data = pd.concat([wsp_data, csv_data.loc[:, "WindSpeed_" + str(k + 1)]], axis=1)
                        wdr_data = pd.concat([wdr_data, csv_data.loc[:, "WindDirection_" + str(k + 1)]], axis=1)
                        th_data = pd.concat([th_data, csv_data.loc[:, "PotentialTemperature_" + str(k + 1)]], axis=1)
                        qv_data = pd.concat([qv_data, csv_data.loc[:, "MixingRatio_" + str(k + 1)]], axis=1)
                    u_data.insert(len(u_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h", calc_u(wsp_data["WindSpeed_" + str(k + 1)], wdr_data["WindDirection_" + str(k + 1)]))
                    u_data[analysis + "_" + str(spinup).zfill(2) + "d15h"] = u_data[analysis + "_" + str(spinup).zfill(2) + "d15h"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                    v_data.insert(len(v_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h", calc_v(wsp_data["WindSpeed_" + str(k + 1)], wdr_data["WindDirection_" + str(k + 1)]))
                    v_data[analysis + "_" + str(spinup).zfill(2) + "d15h"] = v_data[analysis + "_" + str(spinup).zfill(2) + "d15h"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                    wsp_data.rename(columns={"WindSpeed_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                    wdr_data.rename(columns={"WindDirection_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                    th_data.rename(columns={"PotentialTemperature_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                    qv_data.rename(columns={"MixingRatio_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                wsp_data[analysis + "_average"] = wsp_data.filter(regex=analysis, axis=1).mean(axis=1)
                wsp_data[analysis + "_std"] = wsp_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
                wdr_data[analysis + "_average"] = wdr_data.filter(regex=analysis, axis=1).mean(axis=1)
                wdr_data[analysis + "_std"] = wdr_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
                u_data[analysis + "_average"] = u_data.filter(regex=analysis, axis=1).mean(axis=1)
                u_data[analysis + "_std"] = u_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
                v_data[analysis + "_average"] = v_data.filter(regex=analysis, axis=1).mean(axis=1)
                v_data[analysis + "_std"] = v_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
                th_data[analysis + "_average"] = th_data.filter(regex=analysis, axis=1).mean(axis=1)
                th_data[analysis + "_std"] = th_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
                qv_data[analysis + "_average"] = qv_data.filter(regex=analysis, axis=1).mean(axis=1)
                qv_data[analysis + "_std"] = qv_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
            height = str(Decimal(str(sum(height) / len(height))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

            for variable in variables:
                print("Plotting " + variable + " at an altitude of " + height + " m AGL at " + observatory + " on " + date_str)
                graph_name = date_str + "_" + observatory + "_vert-" + str(k + 1).zfill(2) + "_" + variable
                if graph_format == ".pdf":
                    pp = PdfPages(graph_dir + graph_name + graph_format)
                ymin, ymax, iplot = 0, 0, 0

                if variable == "WindSpeed":
                    wrf_data = wsp_data.copy()
                elif variable == "WindDirection":
                    wrf_data = wdr_data.copy()
                elif variable == "u":
                    wrf_data = u_data.copy()
                elif variable == "v":
                    wrf_data = v_data.copy()
                elif variable == "PotentialTemperature":
                    wrf_data = th_data.copy()
                elif variable == "MixingRatio":
                    wrf_data = qv_data.copy()
                x = wrf_data.index


                # Plot all values
                fig = plt.figure(figsize=figsize_setting)
                ax = fig.add_subplot(111, xlabel="Time", ylabel=variables[variable])
                plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                for analysis in analyses:
                    if analysis == "FNL": cmap = plt.get_cmap("Reds")
                    if analysis == "ERA": cmap = plt.get_cmap("Blues")
                    if analysis == "GPV": cmap = plt.get_cmap("Greens")
                    i = 4

                    for spinup in spinups:
                        y = wrf_data.loc[:, analysis + "_" + str(spinup).zfill(2) + "d15h"]
                        if variable == "WindDirection":
                            ax.scatter(x, y, color=cmap(float(i)/24), label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        else:
                            ax.plot(x, y, color=cmap(float(i)/24), label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        i += 1

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=2)
                ax.set_title(date_str + "  approx. " + height + " m AGL")

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + graph_format)
                plt.close()


                # Plot values per analysis data
                for analysis in analyses:
                    fig = plt.figure(figsize=figsize_setting)
                    ax = fig.add_subplot(111, xlabel="Time", ylabel=variables[variable])
                    plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                    if analysis == "FNL": cmap = plt.get_cmap("Reds")
                    if analysis == "ERA": cmap = plt.get_cmap("Blues")
                    if analysis == "GPV": cmap = plt.get_cmap("Greens")
                    i = 4

                    for spinup in spinups:
                        y = wrf_data.loc[:, analysis + "_" + str(spinup).zfill(2) + "d15h"]
                        if variable == "WindDirection":
                            ax.scatter(x, y, color=cmap(float(i)/24), label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        else:
                            ax.plot(x, y, color=cmap(float(i)/24), label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        i += 1

                    ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                    ax.set_title(date_str + "  approx. " + height + " m AGL")

                    if graph_format == ".pdf":
                        pp.savefig()
                    else:
                        plt.savefig(graph_dir + graph_name + "_" + analysis + graph_format)
                    plt.close()


                # Plot averaged values
                fig = plt.figure(figsize=figsize_setting)
                ax = fig.add_subplot(111, xlabel="Time", ylabel=variables[variable])
                plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                for analysis in analyses:
                    if analysis == "FNL": plot_color = "Red"
                    if analysis == "ERA": plot_color = "Blue"
                    if analysis == "GPV": plot_color = "Green"

                    ax.fill_between(x, wrf_data[analysis + "_average"] - wrf_data[analysis + "_std"], wrf_data[analysis + "_average"] + wrf_data[analysis + "_std"], facecolor=plot_color, alpha=0.1)
                    if variable == "WindDirection":
                        ax.scatter(x, wrf_data[analysis + "_average"], color=plot_color, label=analysis+"_average", zorder=20)
                    else:
                        ax.plot(x, wrf_data[analysis + "_average"], linewidth=3, color=plot_color, label=analysis+"_average", zorder=20)

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(date_str + "  approx. " + height + " m AGL")

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + "_average" + graph_format)
                plt.close()


                # Plot averaged values per analysis data
                for analysis in analyses:
                    fig = plt.figure(figsize=figsize_setting)
                    ax = fig.add_subplot(111, xlabel="Time", ylabel=variables[variable])
                    plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                    if analysis == "FNL": plot_color = "Red"
                    if analysis == "ERA": plot_color = "Blue"
                    if analysis == "GPV": plot_color = "Green"

                    ax.fill_between(x, wrf_data[analysis + "_average"] - wrf_data[analysis + "_std"], wrf_data[analysis + "_average"] + wrf_data[analysis + "_std"], facecolor=plot_color, alpha=0.1)
                    if variable == "WindDirection":
                        ax.scatter(x, wrf_data[analysis + "_average"], color=plot_color, label=analysis+"_average", zorder=20)
                    else:
                        ax.plot(x, wrf_data[analysis + "_average"], linewidth=3, color=plot_color, label=analysis+"_average", zorder=20)

                    ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                    ax.set_title(date_str + "  approx. " + height + " m AGL")

                    if graph_format == ".pdf":
                        pp.savefig()
                    else:
                        plt.savefig(graph_dir + graph_name + "_average_" + analysis + graph_format)
                    plt.close()

                if graph_format == ".pdf":
                    pp.close()
