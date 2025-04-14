# radargaugemerging
Python package for statistical merging of radar and gauge precipitation measurements.

## Package contents

The following scripts/modules have been implemented and tested:

| File                                 | Description                                                               |
|--------------------------------------|---------------------------------------------------------------------------|
| collect_radar_gauge_pairs.py         | collect co-located radar-gauge pairs                                      |
| compute_kriged_correction_factors.py | compute Kriging-interpolated radar correction factors                     |
| exporters.py                         | methods for writing output files                                          |
| fit_kriging_model.py                 | fit Kriging model to radar-gauge observation pairs                        |
| importers.py                         | reading radar composites                                                  |
| iterate_kalman_mfb.py                | iterative running of Kalman filter-based mean field bias (MFB) estimation |
| kalman_mfb.py                        | Kalman filter-based model for mean field bias                             |
| radar_archive.py                     | browsing of radar archives                                                |
| regression.py                        | multivariate polynomial regression                                        |
| util.py                              | miscellaneous utility methods                                             |

## Examples

### Run Kalman filter-based MFB

The following example shows how to iteratively apply the mean field bias (MFB) estimator of Chumchean et al. implemented in kalman_mfb.py. Here we assume that the gauge data source is configured to be hourly accumulation and the name of the configuration profile is `config`. To collect gauge-radar pairs between 2017-08-12 15:00-16:00 UTC, we first run

    python collect_gauge_radar_pairs.py 201708121500 201708121600 gaugeradarpairs_201708121600.dat config

This will compute hourly radar rainfall accumulation between the time period and pick the corresponding hourly gauge accumulations ending at 16:00. Using the gauge-radar pair file `gaugeradarpairs_201708121600.dat` created above, we can initialize the MFB estimator by running

    python iterate_kalman_mfb.py 201708121600 gaugeradarpairs_201708121600.dat mfb_state.dat config

This will store the state of the MFB estimator to `mfb_state.dat`. Then we can collect the gauge-radar pair file for the next hour by running

    python collect_gauge_radar_pairs.py 201708121600 201708121700 gaugeradarpairs_201708121700.dat config

Using the previous MFB state, we can then run

    python iterate_kalman_mfb.py 201708121700 gaugeradarpairs_201708121700.dat mfb_state.dat config --prevstatefile mfb_state.dat

This will update the MFB state by using the previously predicted MFB and the most recently observed MFB. This can be repeated to iteratively over successive time steps apply the Kalman filter to produce the MFB estimates.

The logarithmic MFB estimate

$$\displaystyle\beta_t=\frac{1}{n}\sum_{i=1}^n\log_{10}\left(\frac{G_{i,t}}{R_{i,t}}\right)$$

is stored in the dictionary contained in the above state file. The dictionary has the key "corr_factor", whose value can be multiplied with radar-measured rain rates/accumulations to obtain the corrected values.