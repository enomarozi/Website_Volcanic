# Perbaikan integrasi Volcanic Ash Analysis

## Perubahan
- Sidebar disusun ulang sesuai alur klien: Dashboard, Konfigurasi Meteorologi, Konfigurasi Sumber, Model Dispersi, Riwayat, Laporan.
- Router riwayat didaftarkan di `app/main.py`; akses riwayat diarahkan ke login bila sesi tidak aktif.
- Halaman Konfigurasi Meteorologi menyediakan pemilihan NOAA GFS 0.25° / GDAS 0.25°, waktu/cycle, forecast hour, dan koordinat area.
- Adapter NOAA ditambahkan di `app/services/noaa_source.py`: membuat permintaan NOMADS untuk upper-air, surface instant, dan surface accumulated, lalu mencoba konversi GRIB2 ke NetCDF melalui `cfgrib`/`eccodes`.
- Dataset yang berhasil diambil diperiksa memakai validator NetCDF aplikasi sebelum disimpan ke database.
- Halaman Konfigurasi Sumber dan Model Dispersi dibuat dan dihubungkan ke alur simulasi.
- Halaman Laporan mengambil simulasi milik pengguna dari database dan menyediakan ekspor CSV berisi input dan hasil perhitungan.
- Label metadata simulasi tidak lagi mengunci sumber ke ERA5; skrip unduhan ERA5 lama dikeluarkan dari `app/scripts`.
- Daftar dependensi instalasi ditambahkan di `requirements.txt`.

## Instalasi
Aktifkan virtualenv aplikasi lalu jalankan:

```bash
pip install -r requirements.txt
```

Paket konversi GRIB2 membutuhkan `cfgrib` dan `eccodes`, serta dependensi sistem ECCODES sesuai OS. Setelah itu jalankan aplikasi seperti biasa:

```bash
uvicorn app.main:app --reload
```

Buka `/meteorology/config` untuk mencoba akuisisi GFS/GDAS.

## Verifikasi yang sudah dilakukan
- `python -m compileall -q app` berhasil.
- `python -m pytest -q`: 14 pengujian lulus, termasuk pengujian rumus V2 yang sudah ada dan pengujian konstruksi URL NOAA.

## Batas verifikasi
- Lingkungan pemeriksaan ini tidak memiliki `netCDF4`, `cfgrib`, atau `eccodes`, dan tidak digunakan untuk mengunduh GRIB NOAA secara langsung. Karena itu, akuisisi NOAA end-to-end belum dapat dinyatakan lulus di lingkungan ini.
- Endpoint NOMADS, nama variabel/level GRIB, dan hasil konversi harus diuji dari server deployment dengan koneksi internet. Jika struktur GRIB aktual berbeda, adapter perlu disesuaikan sebelum digunakan operasional.
- Data NetCDF lama yang sudah ada di paket tidak dihapus, tetapi tidak dipakai sebagai fallback otomatis oleh adapter GFS/GDAS.
- Input PBL, land-use/topografi ESA WorldCover/GLO-30, dan observasi permukaan belum sepenuhnya dihubungkan sebagai sumber data operasional dalam alur simulasi. Halaman UI tidak berarti fitur-fitur tersebut sudah selesai.
