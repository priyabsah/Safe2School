
# Safe2School ACT — Live Data Version

This version pulls ACT open datasets **directly from the web** (Socrata/GeoJSON URLs) so you don't need local files.
It also allows uploading the ACT population projections Excel file, or you can paste a URL to it.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Default Sources Used
- ACT Daily Public Transport Passenger Journeys (nkxy-abdj) via Socrata API
- ACT School Bus Services (p4rg-3jx2) CSV URL
- ACT Bus Routes (ifm8-78yv) GeoJSON
- Student Distance from Schools (3fd4-5fkk) Socrata v3 JSON
- Census Data for all ACT Schools (8mi2-3658) Socrata v3 JSON
- Park and Ride Locations (sfwt-4uw4) GeoJSON
- Population Projections: upload XLSX in the sidebar or paste a URL
