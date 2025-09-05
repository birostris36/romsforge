# io_utils.py

from netCDF4 import Dataset, MFDataset, num2date, date2num
from datetime import datetime
from typing import Union
from dataclasses import dataclass

def is_netcdf4(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(8)
    return header.startswith(b'CDF\x02') or header.startswith(b'\x89HDF')

def determine_open_mode(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]

    def is_mfdataset_compatible(file_path):
        try:
            with Dataset(file_path) as nc:
                return nc.data_model in ['NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET', 'NETCDF4_CLASSIC']
        except:
            return False

    return 'mf' if all(is_mfdataset_compatible(f) for f in file_list) else 'single'

def parse_time_range(date_input):
    if isinstance(date_input, str):
        t0 = datetime.fromisoformat(date_input)
        return (t0, t0)
    elif isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        t_start = datetime.fromisoformat(date_input[0])
        t_end   = datetime.fromisoformat(date_input[1])
        return (t_start, t_end)
    else:
        raise ValueError("Invalid date input")

@dataclass
class TimeIndex:
    filename: str
    index: int
    datetime: datetime
    raw_value: float

def collect_time_info(input_files, time_var, date_input):
    if isinstance(input_files, str):
        # initial file
        target_date = date_input
        with Dataset(input_files) as nc:
            times = nc.variables[time_var][:]
            units = nc.variables[time_var].units
            tdates = num2date(times, units)
            for i, t in enumerate(tdates):
                if t == target_date:
                    return [TimeIndex(input_files, i, t, times[i])]
        raise ValueError("Target date not found in file.")

    else:
        # boundary files
        t_start, t_end = parse_time_range(date_input)
        time_info = []
        for f in input_files:
            with Dataset(f) as nc:
                times = nc.variables[time_var][:]
                units = nc.variables[time_var].units
                tdates = num2date(times, units)
                for i, t in enumerate(tdates):
                    if t_start <= t <= t_end:
                        time_info.append(TimeIndex(f, i, t, times[i]))
        if not time_info:
            raise ValueError("No valid time steps found in given range.")
        return time_info



def collect_time_info_legacy(input_files, time_var, date_input):
    if isinstance(input_files, str):
        # initial: 단일 파일
        with Dataset(input_files) as nc:
            times = nc.variables[time_var][:]
            units = nc.variables[time_var].units
            tdates = num2date(times, units)
            target_date = datetime.fromisoformat(date_input)
            for i, t in enumerate(tdates):
                if t == target_date:
                    return [(input_files, i, t, times[i])]
            raise ValueError("No matching date in the input file for initdate")

    elif isinstance(input_files, (list, tuple)):
        # boundary: 여러 파일
        t_start, t_end = parse_time_range(date_input)
        time_info = []

        open_mode = determine_open_mode(input_files)
        if open_mode == 'single':
            print("--- [NOTE] Open mode single ---")
            for f in input_files:
                with Dataset(f) as nc:
                    times = nc.variables[time_var][:]
                    units = nc.variables[time_var].units
                    tdates = num2date(times, units)
                    for i, t in enumerate(tdates):
                        if t_start <= t <= t_end:
                            time_info.append((f, i, t, times[i]))
        elif open_mode == 'mf':
            print("--- [NOTE] Open mode multi ---")
            for f in input_files:
                nc = MFDataset(input_files)
                times = nc.variables[time_var][:]
                units = nc.variables[time_var].units
                tdates = num2date(times, units)
            for i, t in enumerate(tdates):
                if t_start <= t <= t_end:
                    time_info.append((None, i, t, times[i]))
        else:
            raise RuntimeError("Unknown open mode")

        assert len(time_info) > 0, "No valid time steps found in OGCM files for given date range."

        return time_info

    else:
        raise ValueError("Invalid input file type: must be str or list")

