import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

year = 2015
targets = [811, 815, 823, 911]
observatory = "Ohashi"
analyses = ["FNL", "ERA", "GPV"]
variables = {"WindSpeed": "Wind Speed [m/s]", "WindDirection": "Wind Direction [°]", "u": "$\it{u}}$ [m/s]", "v": "$\it{v}}$ [m/s]"}

lidar_altitudes = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
sensor_height = 8
rotation_angle = 0.0  # clockwise is positive
wrf_zindex = [2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11]  # zero-based index

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


def calc_u(speed, direction, rotation_angle):
    return - speed * np.sin(np.deg2rad(direction - rotation_angle))

def calc_v(speed, direction, rotation_angle):
    return - speed * np.cos(np.deg2rad(direction - rotation_angle))

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

    work_dir = date_str + "/" + observatory + "/vert/"
    os.makedirs(work_dir + "lidar-wrf_timeseries_plots", exist_ok=True)
    graph_dir = work_dir + "lidar-wrf_timeseries_plots/"
    os.makedirs(work_dir + "statistics", exist_ok=True)
    stat_dir = work_dir + "statistics/"

    data_cols = [0]  # Timestamp (end of interval)
    for k in range(len(lidar_altitudes)):
        data_cols.append( 7 + k * len(lidar_altitudes))  # Wind Speed (m/s)
        data_cols.append(11 + k * len(lidar_altitudes))  # Wind Direction (°)
        if k == 0:
            u_iloc = len(data_cols) - 1
            v_iloc = len(data_cols)
    interval = int((len(data_cols) - 1) / len(lidar_altitudes) + 2)

    obs_file = work_dir + "lidar_obs/" + str(year) + "_" + str(target).zfill(4)[:2] + "_" + str(target).zfill(4)[2:] + "__00_00_00.sta.txt"
    obs_data = pd.read_table(obs_file, header=41, index_col=[0], parse_dates=[0], encoding="shift-jis", usecols=data_cols)

    for k in range(len(lidar_altitudes)):
        obs_data.insert(u_iloc + k * interval, str(lidar_altitudes[k]) + "m u (m/s)", calc_u(obs_data[str(lidar_altitudes[k]) + "m Wind Speed (m/s)"], obs_data[str(lidar_altitudes[k]) + "m Wind Direction (ｰ)"], rotation_angle))
        obs_data.insert(v_iloc + k * interval, str(lidar_altitudes[k]) + "m v (m/s)", calc_v(obs_data[str(lidar_altitudes[k]) + "m Wind Speed (m/s)"], obs_data[str(lidar_altitudes[k]) + "m Wind Direction (ｰ)"], rotation_angle))
        obs_data[str(lidar_altitudes[k]) + "m u (m/s)"] = obs_data[str(lidar_altitudes[k]) + "m u (m/s)"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
        obs_data[str(lidar_altitudes[k]) + "m v (m/s)"] = obs_data[str(lidar_altitudes[k]) + "m v (m/s)"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))

    for k in range(len(lidar_altitudes)):
        height = []
        for analysis in analyses:
            for spinup in spinups:
                csv_file = work_dir + "data/" + date_str + "_" + observatory + "_" + analysis + "_" + str(spinup).zfill(2) + "d15h_vert_TimeSeries_10min.csv"
                csv_data = pd.read_csv(csv_file, header = 0, index_col = [0], parse_dates = [0])
                height.append(csv_data.loc[:, "Height_" + str(wrf_zindex[k] + 1)].mean())
                if analyses.index(analysis) == 0 and spinups.index(spinup) == 0:
                    wsp_data = pd.DataFrame(csv_data.loc[:, "WindSpeed_" + str(wrf_zindex[k] + 1)])
                    wdr_data = pd.DataFrame(csv_data.loc[:, "WindDirection_" + str(wrf_zindex[k] + 1)])
                    u_data = pd.DataFrame(index=wsp_data.index, columns=[])
                    v_data = pd.DataFrame(index=wsp_data.index, columns=[])
                else:
                    wsp_data = pd.concat([wsp_data, csv_data.loc[:, "WindSpeed_" + str(wrf_zindex[k] + 1)]], axis=1)
                    wdr_data = pd.concat([wdr_data, csv_data.loc[:, "WindDirection_" + str(wrf_zindex[k] + 1)]], axis=1)
                u_data.insert(len(u_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h", calc_u(wsp_data["WindSpeed_" + str(wrf_zindex[k] + 1)], wdr_data["WindDirection_" + str(wrf_zindex[k] + 1)], rotation_angle))
                u_data[analysis + "_" + str(spinup).zfill(2) + "d15h"] = u_data[analysis + "_" + str(spinup).zfill(2) + "d15h"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                v_data.insert(len(v_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h", calc_v(wsp_data["WindSpeed_" + str(wrf_zindex[k] + 1)], wdr_data["WindDirection_" + str(wrf_zindex[k] + 1)], rotation_angle))
                v_data[analysis + "_" + str(spinup).zfill(2) + "d15h"] = v_data[analysis + "_" + str(spinup).zfill(2) + "d15h"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                wsp_data.rename(columns={"WindSpeed_" + str(wrf_zindex[k] + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
                wdr_data.rename(columns={"WindDirection_" + str(wrf_zindex[k] + 1): analysis + "_" + str(spinup).zfill(2) + "d15h"}, inplace=True)
            wsp_data[analysis + "_average"] = wsp_data.filter(regex=analysis, axis=1).mean(axis=1)
            wsp_data[analysis + "_std"] = wsp_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
            wdr_data[analysis + "_average"] = wdr_data.filter(regex=analysis, axis=1).mean(axis=1)
            wdr_data[analysis + "_std"] = wdr_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
            u_data[analysis + "_average"] = u_data.filter(regex=analysis, axis=1).mean(axis=1)
            u_data[analysis + "_std"] = u_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
            v_data[analysis + "_average"] = v_data.filter(regex=analysis, axis=1).mean(axis=1)
            v_data[analysis + "_std"] = v_data.filter(regex=analysis, axis=1).std(ddof=0, axis=1)
        wsp_data = pd.concat([wsp_data, obs_data.loc[:, str(lidar_altitudes[k]) + "m Wind Speed (m/s)"]], axis=1)
        wsp_data.rename(columns={str(lidar_altitudes[k]) + "m Wind Speed (m/s)": "LIDAR Obs."}, inplace=True)
        wdr_data = pd.concat([wdr_data, obs_data.loc[:, str(lidar_altitudes[k]) + "m Wind Direction (ｰ)"]], axis=1)
        wdr_data.rename(columns={str(lidar_altitudes[k]) + "m Wind Direction (ｰ)": "LIDAR Obs."}, inplace=True)
        u_data = pd.concat([u_data, obs_data.loc[:, str(lidar_altitudes[k]) + "m u (m/s)"]], axis=1)
        u_data.rename(columns={str(lidar_altitudes[k]) + "m u (m/s)": "LIDAR Obs."}, inplace=True)
        v_data = pd.concat([v_data, obs_data.loc[:, str(lidar_altitudes[k]) + "m v (m/s)"]], axis=1)
        v_data.rename(columns={str(lidar_altitudes[k]) + "m v (m/s)": "LIDAR Obs."}, inplace=True)
        height = str(Decimal(str(sum(height) / len(height))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

        for variable in variables:
            print("Plotting " + variable + " at an altitude of " + str(lidar_altitudes[k] + sensor_height) + " m AGL at " + observatory + " on " + date_str)
            graph_name = date_str + "_" + observatory + "_lidar-wrf_" + str(lidar_altitudes[k] + sensor_height).zfill(3) + "m_" + variable
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
            y = wrf_data.loc[:, "LIDAR Obs."]
            if variable == "WindDirection":
                ax.scatter(x, y, color="black", label="LIDAR Obs.", zorder=20)
            else:
                ax.plot(x, y, color="black", label="LIDAR Obs.", zorder=20)

            ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=2)
            ax.set_title(date_str + "  LIDAR: " + str(lidar_altitudes[k] + sensor_height) + " m AGL,  WRF: approx. " + height + " m AGL")

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
                y = wrf_data.loc[:, "LIDAR Obs."]
                if variable == "WindDirection":
                    ax.scatter(x, y, color="black", label="LIDAR Obs.", zorder=20)
                else:
                    ax.plot(x, y, color="black", label="LIDAR Obs.", zorder=20)

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(date_str + "  LIDAR: " + str(lidar_altitudes[k] + sensor_height) + " m AGL,  WRF: approx. " + height + " m AGL")

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
            y = wrf_data.loc[:, "LIDAR Obs."]
            if variable == "WindDirection":
                ax.scatter(x, y, color="black", label="LIDAR Obs.", zorder=20)
            else:
                ax.plot(x, y, color="black", label="LIDAR Obs.", zorder=20)

            ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
            ax.set_title(date_str + "  LIDAR: " + str(lidar_altitudes[k] + sensor_height) + " m AGL,  WRF: approx. " + height + " m AGL")

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
                y = wrf_data.loc[:, "LIDAR Obs."]
                if variable == "WindDirection":
                    ax.scatter(x, y, color="black", label="LIDAR Obs.", zorder=20)
                else:
                    ax.plot(x, y, color="black", label="LIDAR Obs.", zorder=20)

                ymin, ymax, iplot = set_axes(variable, ymin, ymax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(date_str + "  LIDAR: " + str(lidar_altitudes[k] + sensor_height) + " m AGL,  WRF: approx. " + height + " m AGL")

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + "_" + analysis + "_average" + graph_format)
                plt.close()

            if graph_format == ".pdf":
                pp.close()



            print("Calculating statistics of " + variable + " at an altitude of " + str(lidar_altitudes[k] + sensor_height) + " m AGL at " + observatory + " on " + date_str)
            for analysis in analyses:
                wrf_data.drop(columns=analysis + "_std", inplace=True)
            wrf_data_describe = wrf_data.describe()
            wrf_data_describe.loc["std"] = wrf_data.std(ddof=0)
            wrf_data_describe.loc["std_ratio"] = wrf_data_describe.loc["std"] / wrf_data_describe.loc["std", "LIDAR Obs."]
            wrf_data_stat = wrf_data.corr()
            wrf_data_stat.rename(index={"LIDAR Obs.": "correlation"}, inplace=True)
            wrf_data_stat = pd.concat([wrf_data_describe, wrf_data_stat[-1:]], axis=0)
            wrf_data_stat.loc["bias"] = wrf_data_stat.loc["mean"] - wrf_data_stat.loc["mean", "LIDAR Obs."]
            wrf_data_rmse = wrf_data.copy()
            for col in range(len(wrf_data_rmse.columns)):
                wrf_data_rmse.iloc[:, col] = (wrf_data_rmse.iloc[:, col] - wrf_data_rmse["LIDAR Obs."]) ** 2.0
            wrf_data_stat.loc["rmse"] = wrf_data_rmse.mean() ** 0.5
            wrf_data_centered_rmse = wrf_data.copy()
            for col in range(len(wrf_data_centered_rmse.columns)):
                wrf_data_centered_rmse.iloc[:, col] = (wrf_data_centered_rmse.iloc[:, col] - wrf_data.iloc[:, col].mean() - wrf_data_centered_rmse["LIDAR Obs."] + wrf_data["LIDAR Obs."].mean()) ** 2.0
            wrf_data_stat.loc["centered rmse"] = wrf_data_centered_rmse.mean() ** 0.5
            wrf_data_stat.to_csv(stat_dir + graph_name + "_stat.csv")
