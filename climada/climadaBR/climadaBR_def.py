"""
Define ClimadaBR.
"""
from climada.hazard import *
from climada.entity import *
from climada.engine import *
import numpy as np
from scipy import sparse
import xarray as xr

class ClimadaBR():
    def __init__(self,
                 exp_lp: LitPop = None,
                 impf_set: ImpactFuncSet = None,
                 haz: Hazard = None):
        self.exp_lp = exp_lp
        self.impf_set = impf_set
        self.haz = haz

    def DefineExposures(self, countries, res_arcsec, fin_mode, data_dir = SYSTEM_DIR):
        # using gpw_v4_population_count_rev11_2020_30_sec.tif (NASA)
        self.exp_lp = LitPop.from_countries(countries=countries, res_arcsec=res_arcsec, fin_mode=fin_mode, data_dir = data_dir)
        self.exp_lp.check()
        self.exp_lp.plot_raster()

    def DefineHazards(self, ds, n_ev):

        intensity_sparse = sparse.csr_matrix(ds['intensity'].values)
        fraction_sparse = sparse.csr_matrix(ds['fraction'].values)
        centroids = Centroids.from_lat_lon(ds['latitude'].values, ds['longitude'].values)
        event_id = np.arange(n_ev, dtype=int)
        event_name = ds['event_name'].values.tolist()
        date = ds['event_date'].values
        orig = np.zeros(n_ev, dtype=bool)
        frequency = np.ones(n_ev) / n_ev

        self.haz = Hazard(haz_type='WS',
                    intensity=intensity_sparse,
                    fraction=fraction_sparse,
                    centroids=centroids,  # default crs used
                    units='impact',
                    event_id=event_id,
                    event_name=event_name,
                    date=date,
                    orig=orig,
                    frequency=frequency
        )

        self.haz.check()
        self.haz.centroids.plot()

    def DefineRandomHazards(self):
        lat = np.array([    -22.90685, -23.55052, -12.9714, -8.04728, -3.71722,
            -27.5954, -25.4296, -22.7556, -16.463, -2.81972,
            -9.6658, -12.2628, -8.0512, -22.8119, -3.71722,
            -15.601, -30.0346, -3.10194, -22.3789, -21.7611,
            -13.0166, -4.1008, -2.53073, -28.2639, -20.355,
            -23.8101, -22.9625, -14.8233, -19.0139, -11.4236,
            -23.7122, -21.7611, -22.9056, -23.967, -25.5163,
            -3.7973, -12.2552, -2.897, -14.8628, -1.4485,
            -7.71833, -25.4284, -17.2298, -8.7597, -20.6713,
            -12.9333, -21.6226, -18.7154, -4.3601, -25.9625])

        lon = np.array([    -43.1729, -46.63331, -38.5014, -34.8788, -38.5433,
            -48.548, -49.2713, -41.8787, -39.1523, -40.3097,
            -35.7353, -38.9577, -34.877, -43.1791, -38.5433,
            -38.097, -51.2177, -60.025, -41.778, -41.3307,
            -38.9224, -38.535, -44.3028, -48.6756, -40.2508,
            -45.7033, -42.3656, -40.0638, -39.7496, -37.3623,
            -45.8513, -41.3307, -43.1964, -46.2945, -48.6713,
            -38.5747, -38.9868, -41.1677, -40.8006, -48.5043,
            -34.9128, -49.064, -39.0104, -35.7025, -40.229,
            -38.9995, -41.059, -39.2536, -39.3044, -48.6356])

        # EM NOSSO PROJETO, CADA EVENTO SERA SEU PROPRIO CENTROIDE
        n_cen = 50 # number of centroids
        n_ev = 50 # number of events

        # A INTENSIDADE DOS EVENTOS, NO PROJETO, SERA ESTIMADA POR VALORES DEFINIDOS
        # NAS NOTICIAS, COM APOIO DE LLM. AQUI, GERAMOS RANDOM.
        intensity = sparse.csr_matrix(np.random.random((n_ev, n_cen)))
        fraction = intensity.copy()
        fraction.data.fill(1)

        event_name = []
        for i in range(1,n_ev+1): event_name.append('event_'+str(i))

        event_date = []
        for i in range(1,n_ev+1): event_date.append(721166+i)

        intensity_dense = intensity.toarray()
        fraction_dense = fraction.toarray()

        ds = xr.Dataset(
            {
                'intensity': (['event', 'centroid'], intensity_dense),
                'fraction': (['event', 'centroid'], fraction_dense),
                'event_date': (['event'], event_date)
            },
            coords={
                'latitude': (['centroid'], lat),
                'longitude': (['centroid'], lon),
                'event_name': (['event'], event_name)
            }
        )

        self.DefineHazards(ds, n_ev)

    def AddImpactFunc(self, imp_fun):
        # check if the all the attributes are set correctly
        imp_fun.check()
        imp_fun.plot()

        # add the impact function to an Impact function set
        self.impf_set.append(imp_fun)
        self.impf_set.check()

    def DefineRandomImpactFuncSet(self):
        haz_type = "WS"
        name = "WS Impact Function"
        intensity_unit = "ws impact"


        # provide RANDOM values for the hazard intensity, mdd, and paa
        # AQUI TAMBEM TEMOS QUE DEFINIR COM BASE NOS EVENTOS E COM APOIO DE LLM

        # PARAMETROS QUE IMPACT FUNCTION PRECISA
        # intensity: Intensity values
        # mdd: Mean damage (impact) degree for each intensity (numbers in [0,1])
        # paa: Percentage of affected assets (exposures) for each intensity (numbers in [0,1])

        intensity = np.linspace(0, 100, num=15)
        mdd = np.concatenate((np.array([0]), np.sort(np.random.rand(14))), axis=0)
        paa = np.concatenate((np.array([0]), np.sort(np.random.rand(14))), axis=0)

        imp_fun = ImpactFunc(
            id='WEBSENSORS',
            name=name,
            intensity_unit=intensity_unit,
            haz_type=haz_type,
            intensity=intensity,
            mdd=mdd,
            paa=paa,
        )

        self.impf_set = ImpactFuncSet()

        self.AddImpactFunc(imp_fun)

    def ComputeImpact(self):
        # Get the hazard type and hazard id
        [haz_type] = self.impf_set.get_hazard_types()
        [haz_id] = self.impf_set.get_ids()[haz_type]
        self.exp_lp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
        self.exp_lp.gdf['impf_' + haz_type] = haz_id
        self.exp_lp.gdf

        # Compute impact
        imp = ImpactCalc(self.exp_lp, self.impf_set, self.haz).impact(save_mat=False)  # Do not save the results geographically resolved (only aggregate values)

        imp.plot_raster_eai_exposure()

        print(f"Aggregated average annual impact: {round(imp.aai_agg,0)} $")