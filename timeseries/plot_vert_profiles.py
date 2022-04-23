import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

year = 2015
targets = [811, 815, 823, 911]
observatories = ["Fukuoka", "Ohashi"]
analyses = ["FNL", "ERA", "GPV"]
variables = {"WindSpeed": "Wind Speed [m/s]", "WindDirection": "Wind Direction [°]", "u": "$\it{u}}$ [m/s]", "v": "$\it{v}}$ [m/s]", "PotentialTemperature": "Potential Temperature [K]", "MixingRatio": "Mixing Ratio [kg/kg]"}

lidar_altitudes = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
sensor_height = 8
k_max = 13

graph_format = ".pdf"  # ".png", ".svg", ".pdf", etc.

plt.rcParams["font.family"] = "Arial"
if graph_format == ".svg":
    plt.rcParams["svg.fonttype"] = "none"
if graph_format == ".pdf":
    plt.rcParams["pdf.fonttype"] = 42

figsize_setting = (6, 4)
left_adjust = 0.1
right_adjust = 0.5
bottom_adjust = 0.11
top_adjust = 0.94


def calc_u(speed, direction):
    return - speed * np.sin(np.deg2rad(direction))

def calc_v(speed, direction):
    return - speed * np.cos(np.deg2rad(direction))

def set_axes(variable, xmin, xmax, iplot):
    ax.grid(zorder=0)

    if variable == "WindDirection":
        ax.set_xlim([0, 360])
        ax.set_xticks(np.linspace(0.0, 360.0, 9))
        ax.set_xticks(np.linspace(22.5, 337.5, 8), minor=True)
        ax.grid(which="minor", axis="x", color="lightgray", linestyle="--", linewidth=1)
    elif variable == "WindSpeed":
        if iplot == 0:
            xmin, xmax = ax.get_xlim()
            xmin = 0.0
            xmax = int(xmax) + 1
        ax.set_xlim([xmin, xmax])
    elif variable == "MixingRatio":
        if iplot == 0:
            xmin, xmax = ax.get_xlim()
            xmin = int(xmin * 1000.0) * 0.001
            xmax = (int(xmax * 1000.0) + 1) * 0.001
        ax.set_xlim([xmin, xmax])
    else:
        if iplot == 0:
            xmin, xmax = ax.get_xlim()
            if xmin < 0:
                xmin = int(xmin) - 1
                ax.axvline(x=0, color="black", linewidth=1.0, zorder=10)
            else:
                xmin = int(xmin)
            xmax = int(xmax) + 1
        else:
            if xmin < 0:
                ax.axvline(x=0, color="black", linewidth=1.0, zorder=10)
        ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 300])

    return xmin, xmax, iplot + 1


for target in targets:
    if target == 911:
        spinups = [1, 2, 3, 5, 7, 10, 20, 30, 60]
        markers = ["o", "v", "^", "<", ">", "s", "D", "p", "h", "*"]
    else:
        spinups = [1, 2, 3, 5, 7, 10, 20, 30]
        markers = ["o", "v", "^", "<", ">", "s", "D", "p", "h"]
    date_str = str(year) + "-" + str(target).zfill(4)[:2] + "-" + str(target).zfill(4)[2:]

    for observatory in observatories:
        work_dir = date_str + "/" + observatory + "/vert/"
        os.makedirs(work_dir + "profile_plots", exist_ok=True)
        graph_dir = work_dir + "profile_plots/"

        if observatory == "Ohashi":
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
                obs_data.insert(u_iloc + k * interval, str(lidar_altitudes[k]) + "m u (m/s)", calc_u(obs_data[str(lidar_altitudes[k]) + "m Wind Speed (m/s)"], obs_data[str(lidar_altitudes[k]) + "m Wind Direction (ｰ)"]))
                obs_data.insert(v_iloc + k * interval, str(lidar_altitudes[k]) + "m v (m/s)", calc_v(obs_data[str(lidar_altitudes[k]) + "m Wind Speed (m/s)"], obs_data[str(lidar_altitudes[k]) + "m Wind Direction (ｰ)"]))
                obs_data[str(lidar_altitudes[k]) + "m u (m/s)"] = obs_data[str(lidar_altitudes[k]) + "m u (m/s)"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                obs_data[str(lidar_altitudes[k]) + "m v (m/s)"] = obs_data[str(lidar_altitudes[k]) + "m v (m/s)"].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))

        for k in range(k_max):
            for analysis in analyses:
                for spinup in spinups:
                    csv_file = work_dir + "data/" + date_str + "_" + observatory + "_" + analysis + "_" + str(spinup).zfill(2) + "d15h_vert_TimeSeries_10min.csv"
                    csv_data = pd.read_csv(csv_file, header = 0, index_col = [0], parse_dates = [0])
                    if k == 0 and analyses.index(analysis) == 0 and spinups.index(spinup) == 0:
                        hgt_data = pd.DataFrame(csv_data.loc[:, "Height_" + str(k + 1)])
                        wsp_data = pd.DataFrame(csv_data.loc[:, "WindSpeed_" + str(k + 1)])
                        wdr_data = pd.DataFrame(csv_data.loc[:, "WindDirection_" + str(k + 1)])
                        u_data = pd.DataFrame(index=wsp_data.index, columns=[])
                        v_data = pd.DataFrame(index=wsp_data.index, columns=[])
                        th_data = pd.DataFrame(csv_data.loc[:, "PotentialTemperature_" + str(k + 1)])
                        qv_data = pd.DataFrame(csv_data.loc[:, "MixingRatio_" + str(k + 1)])
                    else:
                        hgt_data = pd.concat([hgt_data, csv_data.loc[:, "Height_" + str(k + 1)]], axis=1)
                        wsp_data = pd.concat([wsp_data, csv_data.loc[:, "WindSpeed_" + str(k + 1)]], axis=1)
                        wdr_data = pd.concat([wdr_data, csv_data.loc[:, "WindDirection_" + str(k + 1)]], axis=1)
                        th_data = pd.concat([th_data, csv_data.loc[:, "PotentialTemperature_" + str(k + 1)]], axis=1)
                        qv_data = pd.concat([qv_data, csv_data.loc[:, "MixingRatio_" + str(k + 1)]], axis=1)
                    u_data.insert(len(u_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k), calc_u(wsp_data["WindSpeed_" + str(k + 1)], wdr_data["WindDirection_" + str(k + 1)]))
                    u_data[analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)] = u_data[analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                    v_data.insert(len(v_data.columns), analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k), calc_v(wsp_data["WindSpeed_" + str(k + 1)], wdr_data["WindDirection_" + str(k + 1)]))
                    v_data[analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)] = v_data[analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)].map(lambda x: float(Decimal(str(x)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)))
                    hgt_data.rename(columns={"Height_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)}, inplace=True)
                    wsp_data.rename(columns={"WindSpeed_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)}, inplace=True)
                    wdr_data.rename(columns={"WindDirection_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)}, inplace=True)
                    th_data.rename(columns={"PotentialTemperature_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)}, inplace=True)
                    qv_data.rename(columns={"MixingRatio_" + str(k + 1): analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(k)}, inplace=True)
                hgt_data[analysis + "_average_" + str(k)] = hgt_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                wsp_data[analysis + "_average_" + str(k)] = wsp_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                wsp_data[analysis + "_std_" + str(k)] = wsp_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)
                wdr_data[analysis + "_average_" + str(k)] = wdr_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                wdr_data[analysis + "_std_" + str(k)] = wdr_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)
                u_data[analysis + "_average_" + str(k)] = u_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                u_data[analysis + "_std_" + str(k)] = u_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)
                v_data[analysis + "_average_" + str(k)] = v_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                v_data[analysis + "_std_" + str(k)] = v_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)
                th_data[analysis + "_average_" + str(k)] = th_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                th_data[analysis + "_std_" + str(k)] = th_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)
                qv_data[analysis + "_average_" + str(k)] = qv_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).mean(axis=1)
                qv_data[analysis + "_std_" + str(k)] = qv_data.filter(regex=analysis, axis=1).filter(regex="_" + str(k), axis=1).std(ddof=0, axis=1)


        for t in range(len(hgt_data)):
            print("Plotting profiles at " + observatory + " on " + str(hgt_data.index[t]))
            for variable in variables:
                graph_name = str(hgt_data.index[t])[:10] + "_" + str(hgt_data.index[t])[11:16].replace(":", "") + "_" + observatory + "_" + variable
                if graph_format == ".pdf":
                    pp = PdfPages(graph_dir + graph_name + graph_format)

                xmin, xmax, iplot = 0, 0, 0

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

                # Plot all values
                fig = plt.figure(figsize=figsize_setting)
                ax = fig.add_subplot(111, xlabel=variables[variable], ylabel="Height Above Ground Level [m]")
                plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                for analysis in analyses:
                    if analysis == "FNL": cmap = plt.get_cmap("Reds")
                    if analysis == "ERA": cmap = plt.get_cmap("Blues")
                    if analysis == "GPV": cmap = plt.get_cmap("Greens")
                    i = 4

                    for spinup in spinups:
                        x = wrf_data.loc[wrf_data.index[t], [analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(j) for j in range(k_max)]]
                        y = hgt_data.loc[hgt_data.index[t], [analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(j) for j in range(k_max)]]
                        ax.plot(x, y, color=cmap(float(i)/24), marker=markers[spinups.index(spinup)], markersize=4, label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        i += 1
                if observatory == "Ohashi":
                    if variable == "WindSpeed" or variable == "WindDirection" or variable == "u" or variable == "v":
                        x_iloc = []
                        for h in lidar_altitudes:
                            if variable == "WindSpeed":
                                x_iloc.append(obs_data.columns.get_loc(str(h) + "m Wind Speed (m/s)"))
                                obs_x = obs_data.iloc[t, x_iloc]
                            elif variable == "WindDirection":
                                x_iloc.append(obs_data.columns.get_loc(str(h) + "m Wind Direction (ｰ)"))
                                obs_x = obs_data.iloc[t, x_iloc]
                            elif variable == "u":
                                x_iloc.append(obs_data.columns.get_loc(str(h) + "m u (m/s)"))
                                obs_x = obs_data.iloc[t, x_iloc]
                            elif variable == "v":
                                x_iloc.append(obs_data.columns.get_loc(str(h) + "m v (m/s)"))
                                obs_x = obs_data.iloc[t, x_iloc]
                        obs_y = [h + sensor_height for h in lidar_altitudes]
                        ax.scatter(obs_x, obs_y, s=60, facecolor="None", edgecolors="black", linewidths=2, label="LIDAR Obs.", zorder=30)

                xmin, xmax, iplot = set_axes(variable, xmin, xmax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=2)
                ax.set_title(hgt_data.index[t])

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + graph_format)
                plt.close()


                # Plot values per analysis data
                for analysis in analyses:
                    fig = plt.figure(figsize=figsize_setting)
                    ax = fig.add_subplot(111, xlabel=variables[variable], ylabel="Height Above Ground Level [m]")
                    plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                    if analysis == "FNL": cmap = plt.get_cmap("Reds")
                    if analysis == "ERA": cmap = plt.get_cmap("Blues")
                    if analysis == "GPV": cmap = plt.get_cmap("Greens")
                    i = 4

                    for spinup in spinups:
                        x = wrf_data.loc[wrf_data.index[t], [analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(j) for j in range(k_max)]]
                        y = hgt_data.loc[hgt_data.index[t], [analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(j) for j in range(k_max)]]
                        ax.plot(x, y, color=cmap(float(i)/24), marker=markers[spinups.index(spinup)], markersize=4, label=str(analysis)+"_"+str(spinup).zfill(2)+"d15h", zorder=20)
                        i += 1
                    if observatory == "Ohashi":
                        if variable == "WindSpeed" or variable == "WindDirection" or variable == "u" or variable == "v":
                            ax.scatter(obs_x, obs_y, s=60, facecolor="None", edgecolors="black", linewidths=2, label="LIDAR Obs.", zorder=30)

                    xmin, xmax, iplot = set_axes(variable, xmin, xmax, iplot)
                    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                    ax.set_title(hgt_data.index[t])

                    if graph_format == ".pdf":
                        pp.savefig()
                    else:
                        plt.savefig(graph_dir + graph_name + "_" + analysis + graph_format)
                    plt.close()


                # Plot averaged values
                fig = plt.figure(figsize=figsize_setting)
                ax = fig.add_subplot(111, xlabel=variables[variable], ylabel="Height Above Ground Level [m]")
                plt.subplots_adjust(left=left_adjust, right=right_adjust, bottom=bottom_adjust, top=top_adjust)

                for analysis in analyses:
                    if analysis == "FNL": plot_color = "Red"
                    if analysis == "ERA": plot_color = "Blue"
                    if analysis == "GPV": plot_color = "Green"

                    x = wrf_data.loc[wrf_data.index[t], [analysis + "_average_" + str(j) for j in range(k_max)]]
                    y = hgt_data.loc[hgt_data.index[t], [analysis + "_" + str(spinup).zfill(2) + "d15h_" + str(j) for j in range(k_max)]]
                    ax.plot(x, y, color=plot_color, marker=markers[0], markersize=4, label=str(analysis)+"_average", zorder=20)

                if observatory == "Ohashi":
                    if variable == "WindSpeed" or variable == "WindDirection" or variable == "u" or variable == "v":
                        ax.scatter(obs_x, obs_y, s=60, facecolor="None", edgecolors="black", linewidths=2, label="LIDAR Obs.", zorder=30)

                xmin, xmax, iplot = set_axes(variable, xmin, xmax, iplot)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, ncol=1)
                ax.set_title(hgt_data.index[t])

                if graph_format == ".pdf":
                    pp.savefig()
                else:
                    plt.savefig(graph_dir + graph_name + "_average" + graph_format)
                plt.close()

                if graph_format == ".pdf":
                    pp.close()


