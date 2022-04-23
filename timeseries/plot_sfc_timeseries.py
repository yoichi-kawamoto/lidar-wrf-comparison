import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

year = 2015
targets = [811, 815, 823, 911]
observatories = ["Fukuoka", "Ohashi"]
analyses = ["FNL", "ERA", "GPV"]
variables = {"SeaLevelPressure": "Sea Level Pressure [hPa]", "Temperature": "Temperature [°C]", "VaporPressure": "Vapor Pressure [hPa]", "MixingRatio": "Mixing Ratio [kg/kg]", "WindSpeed": "Wind Speed [m/s]", "WindDirection": "Wind Direction [°]"}

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
        work_dir = date_str + "/" + observatory + "/sfc/"
        os.makedirs(work_dir + "timeseries_plots", exist_ok=True)
        graph_dir = work_dir + "timeseries_plots/"
        os.makedirs(work_dir + "statistics", exist_ok=True)
        stat_dir = work_dir + "statistics/"

        for variable in variables:
            print("Plotting " + variable + " at " + observatory + " on " + date_str)
            graph_name = date_str + "_" + observatory + "_" + variable
            if graph_format == ".pdf":
                pp = PdfPages(graph_dir + graph_name + graph_format)
            ymin, ymax, iplot = 0, 0, 0

            for analysis in analyses:
                for spinup in spinups:
                    csv_file = work_dir + "/data/" + date_str + "_" + observatory + "_" + analysis + "_" + str(spinup).zfill(2) + "d15h_sfc_TimeSeries_10min.csv"
                    csv_data = pd.read_csv(csv_file, header=0, index_col=[0], parse_dates=[0])
                    if analyses.index(analysis) == 0 and spinups.index(spinup) == 0:
                        wrf_data = pd.DataFrame(csv_data[variables[variable]])
                    else:
                        wrf_data = pd.concat([wrf_data, csv_data[variables[variable]]], axis=1)
                    wrf_data.rename(columns={variables[variable]: analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                wrf_data[analysis + "_average"] = wrf_data.filter(regex=analysis, axis=1).mean(axis=1)
                wrf_data[analysis + "_std"] = wrf_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
            if observatory != "Ohashi":
                obs_file = work_dir + "/jma_obs/" + date_str + "_" + observatory + "_10min.csv"
                obs_data = pd.read_csv(obs_file, header=0, index_col=[0], parse_dates=[0])
                wrf_data = pd.concat([wrf_data, obs_data[variables[variable]]], axis=1)
                wrf_data.rename(columns={variables[variable]: "Obs."}, inplace=True)
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
                        ax.scatter(x, y, color=cmap(float(i)/24), label=str(analysis) + "_" + str(spinup).zfill(2) + "d15h", zorder=20)
                    else:
                        ax.plot(x, y, color=cmap(float(i)/24), label=str(analysis) + "_" + str(spinup).zfill(2) + "d15h", zorder=20)
                    i += 1
            if observatory != "Ohashi":
                y = wrf_data.loc[:, "Obs."]
                if variable == "WindDirection":
                    ax.scatter(x, y, color="black", label="Obs.", zorder=20)
                else:
                    ax.plot(x, y, color="black", label="Obs.", zorder=20)

            ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=2)
            ax.set_title(date_str)

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
                        ax.scatter(x, y, color=cmap(float(i)/24), label=str(analysis) + "_" + str(spinup).zfill(2) + "d15h", zorder=20)
                    else:
                        ax.plot(x, y, color=cmap(float(i)/24), label=str(analysis) + "_" + str(spinup).zfill(2) + "d15h", zorder=20)
                    i += 1
                if observatory != "Ohashi":
                    y = wrf_data.loc[:, "Obs."]
                    if variable == "WindDirection":
                        ax.scatter(x, y, color="black", label="Obs.", zorder=20)
                    else:
                        ax.plot(x, y, color="black", label="Obs.", zorder=20)

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(date_str)

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
                    ax.scatter(x, wrf_data[analysis + "_average"], color=plot_color, label=analysis + "_average", zorder=20)
                else:
                    ax.plot(x, wrf_data[analysis + "_average"], linewidth=3, color=plot_color, label=analysis + "_average", zorder=20)
            if observatory != "Ohashi":
                y = wrf_data.loc[:, "Obs."]
                if variable == "WindDirection":
                    ax.scatter(x, y, color="black", label="Obs.", zorder=20)
                else:
                    ax.plot(x, y, linewidth=2, color="black", label="Obs.", zorder=20)

            ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
            ax.set_title(date_str)

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
                    ax.scatter(x, wrf_data[analysis + "_average"], color=plot_color, label=analysis + "_average", zorder=20)
                else:
                    ax.plot(x, wrf_data[analysis + "_average"], linewidth=3, color=plot_color, label=analysis + "_average", zorder=20)
                if observatory != "Ohashi":
                    y = wrf_data.loc[:, "Obs."]
                    if variable == "WindDirection":
                        ax.scatter(x, y, color="black", label="Obs.", zorder=20)
                    else:
                        ax.plot(x, y, linewidth=2, color="black", label="Obs.", zorder=20)

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(date_str)

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + "_" + analysis + "_average" + graph_format)
                plt.close()

            if graph_format == ".pdf":
                pp.close()


            print("Calculating statistics of " + variable + " at " + observatory + " on " + date_str)
            for analysis in analyses:
                wrf_data.drop(columns=analysis + "_std", inplace=True)
            wrf_data_describe = wrf_data.describe()
            wrf_data_describe.loc["std"] = wrf_data.std(ddof=0)
            if observatory == "Ohashi":
                wrf_data_describe.to_csv(stat_dir + graph_name + "_stat.csv")
            else:
                wrf_data_describe.loc["std_ratio"] = wrf_data_describe.loc["std"] / wrf_data_describe.loc["std", "Obs."]
                wrf_data_stat = wrf_data.corr()
                wrf_data_stat.rename(index={"Obs.": "correlation"}, inplace=True)
                wrf_data_stat = pd.concat([wrf_data_describe, wrf_data_stat[-1:]], axis=0)
                wrf_data_stat.loc["bias"] = wrf_data_stat.loc["mean"] - wrf_data_stat.loc["mean", "Obs."]
                wrf_data_rmse = wrf_data.copy()
                for col in range(len(wrf_data_rmse.columns)):
                    wrf_data_rmse.iloc[:, col] = (wrf_data_rmse.iloc[:, col] - wrf_data_rmse["Obs."]) ** 2.0
                wrf_data_stat.loc["rmse"] = wrf_data_rmse.mean() ** 0.5
                wrf_data_centered_rmse = wrf_data.copy()
                for col in range(len(wrf_data_centered_rmse.columns)):
                    wrf_data_centered_rmse.iloc[:, col] = (wrf_data_centered_rmse.iloc[:, col] - wrf_data.iloc[:, col].mean() - wrf_data_centered_rmse["Obs."] + wrf_data["Obs."].mean()) ** 2.0
                wrf_data_stat.loc["centered rmse"] = wrf_data_centered_rmse.mean() ** 0.5
                wrf_data_stat.to_csv(stat_dir + graph_name + "_stat.csv")
