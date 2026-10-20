import glob
import polars as pl
import pandas as pd


class AISStreamLoader:
    def __init__(self, directory_path, max_days=None):
        self.directory_path = directory_path

        # File ordinati (ais_settimana_1, ais_settimana_2, ...) così possiamo
        # fermarci non appena abbiamo letto abbastanza giorni, invece di
        # leggere sempre tutti i parquet anche per uno smoke test breve
        parquet_files = sorted(glob.glob(f"{directory_path}/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Nessun parquet trovato in {directory_path}")

        collected_frames = []
        seen_days = set()

        for pq_file in parquet_files:
            df_file = (
                pl.scan_parquet(pq_file)
                .with_columns(
                    pl.col("base_date_time").str.to_datetime().alias("_dt_parsed")
                )
                .collect()
            )
            df_file = df_file.with_columns(
                df_file["_dt_parsed"].dt.date().alias("_date_only")
            )
            collected_frames.append(df_file)
            seen_days.update(df_file["_date_only"].unique().to_list())

            # Se ci basta quello che abbiamo già letto, ci fermiamo qui:
            # non leggiamo gli altri file parquet (risparmio reale in I/O)
            if max_days is not None and len(seen_days) >= max_days:
                break

        n_files_read = len(collected_frames)
        full_df = pl.concat(collected_frames)
        del collected_frames

        self.unique_days = sorted(seen_days)
        if max_days is not None:
            self.unique_days = self.unique_days[:max_days]

        print(f"Loader pronto: {len(self.unique_days)} giorni rilevati "
              f"(letti {n_files_read} file su {len(parquet_files)} disponibili)")

        # Pre-suddivide per giorno e converte subito in pandas: get_day()
        # diventa un semplice lookup da dizionario, zero riletture del parquet
        self._day_cache = {}
        for day in self.unique_days:
            df_day = full_df.filter(pl.col("_date_only") == day)
            df_day = df_day.with_columns(
                pl.col("_dt_parsed").alias("base_date_time")
            ).drop(["_dt_parsed", "_date_only"])
            self._day_cache[day] = df_day.to_pandas()

        del full_df

    def get_day(self, day_index):
        target_day = self.unique_days[day_index]
        return self._day_cache[target_day], target_day
