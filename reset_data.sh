#!/bin/bash
set -Eeuo pipefail

APP_DIR="/var/www/html/Website_Volcanic"
DB_FILE="$APP_DIR/volcanic.db"
DATA_DIR="$APP_DIR/data"
BACKUP_DIR="$APP_DIR/backups/clear"

if [[ $EUID -ne 0 ]]; then
    echo "Jalankan script sebagai root."
    exit 1
fi

if [[ ! -d "$APP_DIR" || ! -f "$DB_FILE" ]]; then
    echo "Direktori aplikasi atau database tidak ditemukan."
    exit 1
fi

echo "======================================"
echo " WEBSITE VOLCANIC - CLEAR DATA"
echo "======================================"
echo "DATA YANG AKAN DIHAPUS:"
echo "- Seluruh hasil dan riwayat simulasi"
echo "- Seluruh dataset meteorologi"
echo "- File NetCDF dan cache NOAA"
echo "- Seluruh tile ESA WorldCover"
echo "- Seluruh tile Copernicus DEM"
echo
echo "DATA YANG DIPERTAHANKAN:"
echo "- Akun pengguna"
echo "- Kode aplikasi dan konfigurasi"
echo
read -r -p "Ketik CLEAR untuk melanjutkan: " CONFIRM

if [[ "$CONFIRM" != "CLEAR" ]]; then
    echo "Dibatalkan."
    exit 0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

echo "[1/5] Backup database..."
cp -a "$DB_FILE" "$BACKUP_DIR/volcanic_$TIMESTAMP.db"

echo "[2/5] Menghapus data database..."
sqlite3 "$DB_FILE" <<'SQL'
PRAGMA foreign_keys=ON;
BEGIN;
DELETE FROM simulation_results;
DELETE FROM simulations;
DELETE FROM meteorology_datasets;
COMMIT;
VACUUM;
SQL

echo "[3/5] Menghapus data meteorologi..."
find "$DATA_DIR/meteorology" -maxdepth 1 -type f \
    \( -name "*.nc" -o -name "*.nc4" \) -delete
rm -rf "$DATA_DIR/meteorology/.noaa_cache"

echo "[4/5] Menghapus hasil simulasi..."
find "$DATA_DIR/simulations" -mindepth 1 -maxdepth 1 \
    -exec rm -rf -- {} +

echo "[5/5] Menghapus ESA WorldCover dan Copernicus DEM..."
rm -rf "$DATA_DIR/landcover/ESA_WorldCover_2021"

echo
echo "======================================"
echo " CLEAR DATA BERHASIL"
echo "======================================"
echo "Backup database: $BACKUP_DIR/volcanic_$TIMESTAMP.db"
echo "Data simulasi       : Dihapus"
echo "Data meteorologi    : Dihapus"
echo "Copernicus DEM      : Dihapus"
echo "Akun pengguna       : Dipertahankan"
