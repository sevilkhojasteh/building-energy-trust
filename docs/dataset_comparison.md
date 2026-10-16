# Dataset comparison

| | BDG2 | ASHRAE GEP III | CityLearn |
|---|---|---|---|
| Official URL | http://kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2. https://github.com/buds-lab/building-data-genome-project-2/wiki| https://www.kaggle.com/competitions/ashrae-energy-prediction/overview|https://www.citylearn.net (docs) and https://github.com/citylearn-project/CityLearn (code) |
| Licence + where read | CC BY-SA 4.0 (Kaggle "License" box). | Kaggle Competition Rules, section 7 (Rules tab). Not an open licence: data is shared only with people who accept the rules|MIT (citylearn.net, Cite and contribute section)  |
| Publish code + results? | Yes with attribution (CC BY-SA). Paper link: https://www.nature.com/articles/s41597-020-00712-x | Yes. for any purpose|Yes for the code (MIT). Datasets: no separate licence found; TODO module 6 |
| Redistribute raw data? | Not needed: repo has a download script only | No|Not needed: install with pip and load datasets by name |
| Buildings | 1,636 buildings, 3,053 meters, 19 sites in North America and Europe | over 1,000 buildings at several sites|varies by dataset (one CSV file per building). TODO: check the “Dataset Reference” page |
| Meter types | 8: electricity, chilled water, hot water, steam, gas, water, irrigation, solar | 4: electricity, chilled water, steam, hot water|not meters. The building files contain loads (non-shiftable, hot water, cooling, heating), solar generation, indoor temperature and humidity, and occupancy |
| Time span | 2016-01-01 to 2017-12-31 (2 full years) | 3 years of hourly data. Not checked.|varies by dataset; check “Dataset Reference” |
| Resolution | Hourly (17,544 values per meter) | Hourly|check the “Data Unit Contract” page (the timestep) |
| Timezone | 6 ->US/Eastern,Europe/London,US/Mountain,US/Pacific,US/Central,Europe/Dublin | Timezone: Not checked.|files use month/hour/day type, not full timestamps. TODO: confirm |
| Weather | timestamp,siteID,airTemperature,cloudCoverage,dewTemperature,precipDepth1HR,precipDepth6HR,seaLvlPressure,windDirection,windSpeed| Weather: Yes, weather_train.csv and weather_test.csv: air temperature, dew temperature, cloud cover, train, pressure, wind direction, wind speed|yes: temperature, humidity, solar irradiance, plus 3 predicted (forecast) columns. Can be TMY (typical year) or AMY (actual year) |
| Metadata | metadata.csv: building id, site, primary use (+ sub-use), floor area, lat/lng, timezone, which meters exist, industry, heating type, year built, floors, occupants, EUI, Energy Star / LEED ratings| Metadata: Yes, building_metadata.csv: site, primary use, floor area, year built, number of floors|a JSON schema file per scenario, plus carbon intensity and electricity price files |
| Size | 747 MB, 18 CSV files (Kaggle version 3) | 2.61 GB, 6 CSV files|small, installed with pip; check the installation page |
| Known issues | Raw files have many missing values and zeros (e.g. one chilled water column is 67% empty). Cleaned versions exist (*_cleaned.csv). More outliers have been cleaned using Twitter AnomalyDetection R library (https://github.com/twitter/AnomalyDetection), zero readings longer than 24 continuous hours are removed, and zero readings in electricity meters are removed | Site 0 electricity readings are in kBTU, not kWh. Gaps in meter readings. Real measurement errors. test.csv has no meter readings (no labels)|loads can come from EnergyPlus simulation or from real meters. You must check which for the dataset you use, and say “simulated” if it is |
| Citation | Miller, C. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). doi:10.1038/s41597-020-00712-x | Howard, A., Balbach, C., Miller, C., Haberl, J., Gowri, K., Dane, S. ASHRAE - Great Energy Predictor III. Kaggle, 2019. https://www.kaggle.com/competitions/ashrae-energy-prediction|TODO: see the “References” or “Publications” page |
| Relation to others | ASHRAE GEP III used a subset: 2,380 meters from 1,449 buildings | Subset of BDG2|not related. It’s a control environment for module 6, not a forecasting dataset |

## Decision
- Dataset and meter type I choose:
  - BDG2
  - Electricity, because this is the most common meter.
- Why:
  - The second dataset restricts the dataset, declaring that you can not share this dataset with anyone (unless the third party accepts the competition rules), and BDG2 is a open licence dataset.
  - with BDG2 I can train on 2016 and test on 2017, on the other hand, the ASHRAE dataset only train.csv is available due to the nature of the competition, and I need to split the train dataset for training and testing, and I can not test the model for every season.
  - BDG2 has many buildings from 19 different sites, so in module 4 I can treat each site as one client that keeps its data private. Because the sites differ in climates, countries, building types, I can test ___ and check ___ on a site the model never saw in training.
- What I still need to check: 
    - raw or cleaned files? If cleaned, report it in the README as a limitation
    - Are timestamps local?
    - how many of the stated buildings have electricity meter

## Differences
- BDG2 vs ASHRAE in one line: 
  - both allow research use and publishing results. BDG2 is under an open licence (CC BY-SA) with 2 years of data and 8 meter types. ASHRAE is a subset of the same buildings, under stricter competition rules and with only 4 meter types.
