import csv
import sys

sys.path.append('Redacted code')

import numpy as np
import scipy.stats
import datetime
import psycopg2 as pg



from sql_classes_modifiedmedicationquery import load_encounter
from dateutil.relativedelta import relativedelta
from Temperatures import Temperatures
from Sbps import Sbps
from HeartRates import HeartRates
from Gcss import Gcss
from Ventilations import Ventilations
from Resps import Resps
from Spo2s import Spo2s
from Creatinines import Creatinines
from Gfrs import Egfrs
from Cultures import Cultures
from Lactates import Lactates
from Platelets import Platelets
from TotalBilirubins import TotalBilirubins
from WhiteBloodCells import WhiteBloodCells
from Vasopressors import Vasopressors
from Shockindexes import Shockindexes
from Antibiotics import Antibiotics
from operator import attrgetter
from Lactates_result_time import Lactates_result_time
from WhiteBloodCells_result_time import WhiteBloodCells_result_time
from Mortality_or_hospice import Mortality_or_hospice
from FourQAD import FourQad

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)       #reading mean and std error
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)  #ppf finds the probability of normal distribution value #se*ppf
    return str(round(m,1)) + ' (' + str(round(m-h,1)) + ', ' + str(round(m+h,1)) + ')'

def median_25th_75th_percentile(data):
    return str(np.percentile(data, 50)) + ' (' + str(np.percentile(data, 25))  + ', ' + str(np.percentile(data, 75))  + ')'


class Patient:
    def __init__(self, hospitalaccountid, encounterid, esrd, complaint_dms, fatigue_or_ams, majorcomorbidity, pre_ed_infection, bacterialsymptomcomplex, cad_, chf_, ckd_, copd_, cva_, diabetes_, liver_, pneumonia_, uti_, abdominal_, skin_, other_, unknown_):
        self.hospitalaccountid = hospitalaccountid
        self.encounterid = encounterid
        self.qad = qad
        self.end_stage_renal_disease = esrd
        self.fatigue_or_ams = fatigue_or_ams
        self.majorcomorbidity = majorcomorbidity
        self.pre_ed_infection = pre_ed_infection
        self.bacterialsymptomcomplex = bacterialsymptomcomplex
        self.pneumonia = pneumonia_
        self.uti = uti_
        self.abdominal = abdominal_
        self.skin = skin_
        self.other = other_
        self.unknown = unknown_
        self.cad = cad_
        self.chf = chf_
        self.ckd = ckd_
        self.copd = copd_
        self.cva = cva_
        self.diabetes = diabetes_
        self.liver_disease = liver_
        conn = pg.connect(database="epic_screened")
        cur = conn.cursor()
        Encounter = load_encounter(cur)
        self.triagetime = Encounter(self.hospitalaccountid).ed_triage_time_for_account(self.hospitalaccountid)  #sql_classes_modifiedmedicationquery ENCOUNTER class is called after which method to fetch triage time gets called
        self.birthdts = (Encounter(self.hospitalaccountid).patient.birthdts)
        self.age = relativedelta(self.triagetime, self.birthdts).years
        self.race = Encounter(self.hospitalaccountid).race()
        self.mortality = Mortality_or_hospice(self.hospitalaccountid) #different py file
        if self.race == 'White or Caucasian' or self.race == 'White':
            self.white_race = 1
        else:
            self.white_race = 0
        self.sex = Encounter(self.hospitalaccountid).sex()
        if self.sex == 'Male':
            self.malesex = 1
        else:
            self.malesex = 0
        self.ed_in = Encounter(hospitalaccountid).ed_interval.start
        self.ed_out = Encounter(hospitalaccountid).ed_interval.end
        self.predictor_time = self.ed_in + datetime.timedelta(hours=3)
        self.missing_predictor = False
        self.sbps = Sbps(self.hospitalaccountid, True)
        self.hrs = HeartRates(self.hospitalaccountid, True)
        self.shock_indexes = Shockindexes(self.hospitalaccountid, True)
        self.resps = Resps(self.hospitalaccountid, True)
        self.spo2s = Spo2s(self.hospitalaccountid, True)
        self.temps = Temperatures(self.hospitalaccountid, True)
        self.gcss = Gcss(self.hospitalaccountid, True)
        self.continuous_clipped_processed_vital_list = []
        self.continuous_unprocessed_vital_list = []

        self.one_hr_full_model_score = None
        self.one_hr_vs_model_score = None
        self.one_hr_qsofa_score = None
        self.three_hr_full_model_score = None
        self.three_hr_vs_model_score = None
        self.three_hr_qsofa_score = None

        self.sbp_triage = self.clip_vital(Sbps(self.hospitalaccountid, False).get_ed_triage_vital(), None, 100)  #passing lower and upper value
        #Here I used all vitals, not just ED vitals to get the vital closest in time to the ED_in time. If I limited
        #to ED vital only, I may miss some vitals recorded shortly before the official ED_in time.
        self.si_triage = self.clip_vital(Shockindexes(self.hospitalaccountid, False).get_ed_triage_vital(), 0.65, 2.5)
        self.resp_triage = self.clip_vital(Resps(self.hospitalaccountid, False).get_ed_triage_vital(), 16, 30)
        self.spo2_triage = self.clip_vital(Spo2s(self.hospitalaccountid, False).get_ed_triage_vital(), 90, None)
        self.temp_high_triage = self.clip_vital(Temperatures(self.hospitalaccountid, False).get_ed_triage_vital(), 97, 104)
        self.temp_low_triage = self.clip_vital(Temperatures(self.hospitalaccountid, False).get_ed_triage_vital(), 93, 97)
        self.gcs_triage = Gcss(self.hospitalaccountid, False).get_ed_triage_vital()



        if Gcss(self.hospitalaccountid, False).get_ed_triage_vital() is None:
            if complaint_dms == 0: #assigns gcs = 15 if altered mental status label is false on redcap
                self.gcss.add_gcs_measurement(Patient.GenericVital(float(15), self.ed_in))
                self.gcs_triage = Patient.GenericVital(float(15), self.ed_in)
            elif complaint_dms == 1:
                self.gcss.add_gcs_measurement(Patient.GenericVital(float(14), self.ed_in))
                self.gcs_triage = Patient.GenericVital(float(14), self.ed_in)
            else:
                print(self.hospitalaccountid, "missing GCS trigger ")
                self.missing_predictor = True


        if self.sbps.get_ed_triage_vital() is None or self.shock_indexes.get_ed_triage_vital() is None or \
                self.resps.get_ed_triage_vital() is None or self.spo2s.get_ed_triage_vital() is None \
                or self.temps.get_ed_triage_vital() is None:

            print(self.hospitalaccountid, "missingothervital trigger")

            self.missing_predictor = True


        self.timelist = list(dict.fromkeys(self.sbps.get_times() + \
                                           self.shock_indexes.get_times() + self.resps.get_times() \
                                           + self.spo2s.get_times() + self.temps.get_times() + self.gcss.get_times()))
        self.timelist.sort()

        for time in self.timelist:

            sbp_exp_clipped = self.clip_vital(self.sbps.get_most_recent_exp_vital_given_time(time), None, 100)
            si_exp_clipped = self.clip_vital(self.shock_indexes.get_most_recent_exp_vital_given_time(time), 0.65, 2.5)
            temp_low_clipped = self.clip_vital(self.temps.get_min_vital_up_until_time(time), 93, 97)
            temp_high_clipped = self.clip_vital(self.temps.get_max_vital_up_until_time(time), 97, 104)
            resp_exp_clipped = self.clip_vital(self.resps.get_most_recent_expresp_given_time(time), 16, 30)
            spo2_exp_clipped = self.clip_vital(self.spo2s.get_most_recent_exp_vital_given_time(time), 90, None)
            gcs_min = self.gcss.get_min_vital_up_until_time(time)

            sbp_ = self.sbps.get_vital_given_time(time)
            si_ = self.shock_indexes.get_vital_given_time(time)
            temp_ = self.temps.get_vital_given_time(time)
            resp_ = self.resps.get_vital_given_time(time)
            spo2_ = self.spo2s.get_vital_given_time(time)
            gcs_ = self.gcss.get_vital_given_time(time)

            self.continuous_unprocessed_vital_list.append(Patient.ContinuousVital(time, sbp_, si_, resp_, spo2_, temp_, gcs_))

            if sbp_exp_clipped is not None and si_exp_clipped is not None and temp_low_clipped is not None and \
                    temp_high_clipped is not None and resp_exp_clipped is not None and spo2_exp_clipped is not None and \
                    gcs_min is not None:
                self.continuous_clipped_processed_vital_list.append(Patient.ContinuousProcessedVital(time, sbp_exp_clipped.get_value(), si_exp_clipped.get_value(), \
                                                                                                     resp_exp_clipped.get_value(), spo2_exp_clipped.get_value(), temp_high_clipped.get_value(), temp_low_clipped.get_value(), \
                                                                                                     gcs_min.get_value()))

        self.ventilations = Ventilations(self.hospitalaccountid)
        self.creatinines = Creatinines(hospitalaccountid)
        self.gfrs = Egfrs(hospitalaccountid)
        self.lactates = Lactates(hospitalaccountid)
        self.platelets = Platelets(hospitalaccountid)
        self.tbilis = TotalBilirubins(hospitalaccountid)
        self.cultures = Cultures(hospitalaccountid)
        self.wbcs = WhiteBloodCells(hospitalaccountid)
        self.wbc_result_time = WhiteBloodCells_result_time(hospitalaccountid)
        self.lactate_result_time = Lactates_result_time(hospitalaccountid)
        self.antibiotics = Antibiotics(hospitalaccountid)

        self.culture_dysfunction = False
        self.vaso_dysfunction = False
        self.respiratory_dysfunction = False
        self.creatinine_dysfunction = False
        self.gfr_dysfunction = False
        self.platelet_dysfunction = False
        self.tbili_dysfunction = False
        self.lactate_dysfunction = False
        self.rhee_sepsis = 0
        self.vasopressors = Vasopressors(hospitalaccountid)
        self.qsofa = 0
        self.full_model_score = 0
        self.vs_model_score = 0

        try:
            self.sofa = Encounter(self.hospitalaccountid).sofa_composite()
        except Exception as e:
            if str(e) == 'Insufficent GCS for component':
                if self.gcs_triage is not None:
                    if self.gcs_triage.get_value() == 15:
                        self.sofa = Encounter(self.hospitalaccountid).sofa_resp() + Encounter(self.hospitalaccountid).sofa_cardiac() + Encounter(self.hospitalaccountid).sofa_coagulation() + Encounter(self.hospitalaccountid).sofa_liver() + Encounter(self.hospitalaccountid).sofa_renal()
                    elif self.gcs_triage.get_value() == 14:
                        self.sofa = Encounter(self.hospitalaccountid).sofa_resp() + Encounter(self.hospitalaccountid).sofa_cardiac() + Encounter(self.hospitalaccountid).sofa_coagulation() + Encounter(self.hospitalaccountid).sofa_liver() + Encounter(self.hospitalaccountid).sofa_renal() + 1
                    else:
                        self.sofa = None
                else:
                    self.sofa = None
            else:
                print(str(e))
                self.sofa = None

        if self.creatinines.min_lab() is not None:
            for cr in self.creatinines.get_creatinine_labs():
                if cr.get_value() >= 2 * self.creatinines.min_lab().get_value() and \
                        (cr.get_time() - self.ed_in).total_seconds() <= 172800 and\
                        self.end_stage_renal_disease == False:
                    self.creatinine_dysfunction = True

        if self.gfrs.max_lab() is not None:
            for gf in self.gfrs.get_egfr_labs():
                if gf.get_value() <= 0.5 * self.gfrs.max_lab().get_value() and \
                        (gf.get_time() - self.ed_in).total_seconds() <= 172800\
                        and self.end_stage_renal_disease == False:
                    self.gfr_dysfunction = True

        if self.platelets.max_lab() is not None:
            for pl in self.platelets.get_platelet_labs():
                if self.platelets.max_lab().get_value() >= 100 > pl.get_value()\
                        and pl.get_value() <= 0.5 * self.platelets.max_lab().get_value()\
                        and (pl.get_time() - self.ed_in).total_seconds() <= 172800:
                    self.platelet_dysfunction = True

        if self.tbilis.min_lab() is not None:
            for tb in self.tbilis.get_totalbilirubin_labs():
                if tb.get_value() >= 2 and tb.get_value() >= 2 * self.tbilis.min_lab().get_value() \
                        and (tb.get_time() - self.ed_in).total_seconds() <= 172800:
                    self.tbili_dysfunction = True

        if len(self.lactates.get_lactate_labs()) > 0:
            for la in self.lactates.get_lactate_labs():
                if la.get_value() >= 2 and (la.get_time() - self.ed_in).total_seconds() <= 172800:
                    self.lactate_dysfunction = True

        if len(self.cultures.get_cultures()) > 0:
            for cu in self.cultures.get_cultures():
                if (cu.get_time() - self.ed_out).total_seconds() <= 86400:
                    self.culture_dysfunction = True

        self.qad = FourQad(self.ed_in, self.mortality, self.antibiotics)


        if len(self.vasopressors.get_vasopressors()) > 0:
            for va in self.vasopressors.get_vasopressors():
                if (va.get_time() - self.ed_in).total_seconds() <= 172800:
                    self.vaso_dysfunction = True

        if len(self.ventilations.get_ventilations()) > 0:
            for ve in self.ventilations.get_ventilations():
                if (ve.get_time() - self.ed_in).total_seconds() <= 172800:
                    self.respiratory_dysfunction = True


        if (self.culture_dysfunction and self.qad) and (self.vaso_dysfunction or
                                                        self.respiratory_dysfunction
                                                        or self.creatinine_dysfunction
                                                        or self.gfr_dysfunction or
                                                        self.platelet_dysfunction or
                                                        self.tbili_dysfunction or
                                                        self.lactate_dysfunction):
            self.rhee_sepsis = 1

        if self.gcs_triage is not None and self.sbp_triage is not None and self.resp_triage is not None:
            if self.gcs_triage.get_value() < 15:
                self.qsofa += 1
            if self.sbp_triage.get_value() <= 100:
                self.qsofa += 1
            if self.resp_triage.get_value() >= 22:
                self.qsofa += 1

    def __eq__(self, other):
        return self.hospitalaccountid == other.hospitalaccountid

    def get_qsofa(self):
        return self.qsofa
    def get_sofa(self):
        return self.sofa

    def get_pneumonia(self):
        return self.pneumonia
    def get_uti(self):
        return self.uti
    def get_abdominal(self):
        return self.abdominal
    def get_skin(self):
        return self.skin
    def get_other(self):
        return self.other
    def get_unknown(self):
        return self.unknown

    def get_cad(self):
        return self.cad
    def get_chf(self):
        return self.chf
    def get_ckd(self):
        return self.ckd
    def get_copd(self):
        return self.copd
    def get_cva(self):
        return self.cva
    def get_diabetes(self):
        return self.diabetes
    def get_liver_disease(self):
        return self.liver_disease
    def get_triage_time(self):
        return self.triagetime
    def get_edin(self):
        return self.ed_in
    def set_full_model_score(self, score):
        self.full_model_score = score
    def get_full_model_score(self):
        return self.full_model_score
    def set_vs_model_score(self, score):
        self.vs_model_score = score
    def get_vs_model_score(self):
        return self.vs_model_score

    def set_1hr_full_model_score(self, score):
        self.one_hr_full_model_score = score
    def get_1hr_full_model_score(self):
        return self.one_hr_full_model_score
    def set_1hr_vs_model_score(self, score):
        self.one_hr_vs_model_score = score
    def get_1hr_vs_model_score(self):
        return self.one_hr_vs_model_score
    def set_1hr_qsofa_score(self, score):
        self.one_hr_qsofa_score = score
    def get_1hr_qsofa_score(self):
        return self.one_hr_qsofa_score

    def set_3hr_full_model_score(self, score):
        self.three_hr_full_model_score = score
    def get_3hr_full_model_score(self):
        return self.three_hr_full_model_score
    def set_3hr_vs_model_score(self, score):
        self.three_hr_vs_model_score = score
    def get_3hr_vs_model_score(self):
        return self.three_hr_vs_model_score
    def set_3hr_qsofa_score(self, score):
        self.three_hr_qsofa_score = score
    def get_3hr_qsofa_score(self):
        return self.three_hr_qsofa_score


#cha defining functions for SRI computation for 2 hrs
    def set_2hr_full_model_score(self, score):
        self.three_hr_full_model_score = score
    def get_2hr_full_model_score(self):
        return self.three_hr_full_model_score
    def set_2hr_vs_model_score(self, score):
        self.three_hr_vs_model_score = score
    def get_2hr_vs_model_score(self):
        return self.three_hr_vs_model_score
    def set_2hr_qsofa_score(self, score):
        self.three_hr_qsofa_score = score
    def get_2hr_qsofa_score(self):
        return self.three_hr_qsofa_score

    def set_4hr_full_model_score(self, score):
        self.three_hr_full_model_score = score
    def get_4hr_full_model_score(self):
        return self.three_hr_full_model_score
    def set_4hr_vs_model_score(self, score):
        self.three_hr_vs_model_score = score
    def get_4hr_vs_model_score(self):
        return self.three_hr_vs_model_score
    def set_4hr_qsofa_score(self, score):
        self.three_hr_qsofa_score = score
    def get_4hr_qsofa_score(self):
        return self.three_hr_qsofa_score
#cha ending the function definition for 2 and 4 hours SRI computation

    def get_hospital_account_id(self):
        return self.hospitalaccountid
    def get_patient_encounter_id(self):
        return self.encounterid
    def get_missing_predictor(self):
        return self.missing_predictor
    def get_sbp_triage_clipped_value(self):
        return self.sbp_triage.get_value()
    def get_temp_high_triage_clipped_value(self):
        return self.temp_high_triage.get_value()
    def get_temp_low_triage_clipped_value(self):
        return self.temp_low_triage.get_value()
    def get_resp_triage_clipped_value(self):
        return self.resp_triage.get_value()
    def get_spo2_triage_clipped_value(self):
        return self.spo2_triage.get_value()
    def get_shock_index_triage_clipped_value(self):
        return self.si_triage.get_value()
    def get_gcs_triage_value(self):
        return self.gcs_triage.get_value()
    def get_sbp_triage_unclipped_value(self):
        return self.sbps.get_ed_triage_vital().get_value()
    def get_hr_triage_unclipped_value(self):
        return self.hrs.get_ed_triage_vital().get_value()
    def get_shock_index_triage_unclipped_value(self):
        return self.shock_indexes.get_ed_triage_vital().get_value()
    def get_resp_triage_unclipped_value(self):
        return self.resps.get_ed_triage_vital().get_value()
    def get_spo2_triage_unclipped_value(self):
        return self.spo2s.get_ed_triage_vital().get_value()
    def get_temp_triage_unclipped_value(self):
        return self.temps.get_ed_triage_vital().get_value()
    def get_age(self):
        return self.age
    def get_male_sex(self):
        return self.malesex
    def get_white_race(self):
        return self.white_race
    def get_bacterial_symptom_complex(self):
        return self.bacterialsymptomcomplex
    def get_fatigue_ams(self):
        return self.fatigue_or_ams
    def get_major_comorbidity(self):
        return self.majorcomorbidity
    def get_pre_ed_infection(self):
        return self.pre_ed_infection
    def get_rhee_sepsis(self):
        return self.rhee_sepsis
    def get_first_antibiotic_time(self):
        if len(self.antibiotics.get_times()) > 0:
            return min(self.antibiotics.get_times())
        else:
            return 'No antibiotics'
    def get_first_wbc(self):
        return self.wbc_result_time.first_result()
    def get_first_lactate(self):
        return self.lactate_result_time.first_result()
    def get_first_creatinine(self):
        return self.creatinines.first_lab()
    def get_first_platelet(self):
        return self.platelets.first_lab()
    def get_mortality(self):
        return self.mortality.get_mortality()

    def get_culture_dysfunction(self):
        return self.culture_dysfunction
    def get_qad_dysfunction(self):
        return self.qad


    def get_vaso_dysfunction(self):
        return self.vaso_dysfunction
    def get_respiratory_dysfunction(self):
        return self.respiratory_dysfunction
    def get_creatinine_dysfunction(self):
        return self.creatinine_dysfunction
    def get_gfr_dysfunction(self):
        return self.gfr_dysfunction
    def get_platelet_dysfunction(self):
        return self.platelet_dysfunction
    def get_tbili_dysfunction(self):
        return self.tbili_dysfunction
    def get_lactate_dysfunction(self):
        return self.lactate_dysfunction


    def get_processed_vital_list(self):
        return sorted(self.continuous_clipped_processed_vital_list, key=attrgetter("time"))
    def get_processed_vital_set_closest_to_given_time(self, time): #returns the index of closest vital set to a given time and the time difference to the target time.
        valueexists = False
        for vitalset in self.continuous_clipped_processed_vital_list:
            if not valueexists:
                closest_vital_set = vitalset
                min_time_difference = abs((vitalset.get_time() - time).total_seconds())
                valueexists = True
            if valueexists:
                if abs((vitalset.get_time() - time).total_seconds()) < min_time_difference:
                    closest_vital_set = vitalset
                    min_time_difference = abs((vitalset.get_time() - time).total_seconds())
        return [closest_vital_set, min_time_difference]

    def get_unprocessed_vital_list(self):
        return sorted(self.continuous_unprocessed_vital_list, key=attrgetter("time"))

    def clip_vital(self, vital, lower, upper):
        if vital is not None:
            value = vital.get_value()
            time = vital.get_time()
            if upper is not None:
                if value >= upper:
                    value = upper
            if lower is not None:
                if value <= lower:
                    value = lower
            return Patient.GenericVital(value,time)
        else:
            return None

    class GenericVital:
        def __init__(self, value, time):
            self.value = value
            self.time = time
        def get_value(self):
            return self.value
        def get_time(self):
            return self.time

    class ContinuousProcessedVital:
        def __init__(self, time, sbp_exp_cont, si_exp_cont, resp_exp_cont, spo2_exp_cont, temp_high_cont, temp_low_cont, gcs_min_cont):
            self.time = time
            self.sbp_exp_cont = sbp_exp_cont
            self.si_exp_cont = si_exp_cont
            self.resp_exp_cont = resp_exp_cont
            self.spo2_exp_cont = spo2_exp_cont
            self.temp_high_cont = temp_high_cont
            self.temp_low_cont = temp_low_cont
            self.gcs_min_cont = gcs_min_cont

        def get_time(self):
            return self.time
        def get_sbp_exp_cont(self):
            return self.sbp_exp_cont
        def get_si_exp_cont(self):
            return self.si_exp_cont
        def get_resp_exp_cont(self):
            return self.resp_exp_cont
        def get_spo2_exp_cont(self):
            return self.spo2_exp_cont
        def get_temp_high_cont(self):
            return self.temp_high_cont
        def get_temp_low_cont(self):
            return self.temp_low_cont
        def get_gcs_min_cont(self):
            return self.gcs_min_cont

    class ContinuousVital:
        def __init__(self, time, sbp_, si_, resp_, spo2_, temp_, gcs_):
            self.time = time
            self.sbp_cont = sbp_
            self.si_cont = si_
            self.resp_cont = resp_
            self.spo2_cont = spo2_
            self.temp_cont = temp_
            self.gcs_cont = gcs_

        def get_time(self):
            return self.time
        def get_sbp_cont(self):
            return self.sbp_cont
        def get_si_cont(self):
            return self.si_cont
        def get_resp_cont(self):
            return self.resp_cont
        def get_spo2_cont(self):
            return self.spo2_cont
        def get_temp_cont(self):
            return self.temp_cont
        def get_gcs_cont(self):
            return self.gcs_cont

patientlist = []
index = 1
with open('Redacted code', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        hospitalaccountid = int(row['hospitalaccountid'])
        encounterid = int(row['patientencounterid'])
        birthdate = row['birthdts']
        edin = row['ed_in']
        recordid = row['record_id']
        last4mrn = row['mrn']
        edadmittime = row['ed_admit_date_time']
        encountercomplete = row['encounter_info_complete']
        pre_arrival_cfi = int(row['pre_arrival_cfi'])
        infection_source_pta = row['infection_source_pta']
        abx_pta = row['abx_pta']
        vp_pta = row['vp_pta']
        complaint_dms = int(row['complaint_dms'])
        fever_pta = int(row['fever_pta'])
        constitutional = int(row['constitutional'])
        p_symp_complexes = row['p_symp_complexes']
        pmh_cancer_type___0 = int(row['pmh_cancer_type___0'])
        pmh_cancer_type___1 = row['pmh_cancer_type___1']
        pmh_cancer_type___2 = row['pmh_cancer_type___2']
        pmh_cancer_type___3 = row['pmh_cancer_type___3']
        pmh_cancer_type___4 = row['pmh_cancer_type___4']
        pmh_cad = int(row['pmh_cad'])
        pmh_chf =int(row['pmh_chf'])
        pmh_respiratory_illness = int(row['pmh_respiratory_illness'])
        pmh_connective = int(row['pmh_connective'])
        pmh_cva = int(row['pmh_cva'])
        pmh_dm_both = int(row['pmh_dm_both'])
        pmh_immunocompromised = int(row['pmh_immunocompromised'])
        pmh_immunocomp_type___1 = row['pmh_immunocomp_type___1']
        pmh_immunocomp_type___2 = row['pmh_immunocomp_type___2']
        pmh_immunocomp_type___3 = row['pmh_immunocomp_type___3']
        pmh_immunocomp_type___4 = row['pmh_immunocomp_type___4']
        pmh_ivdu = int(row['pmh_ivdu'])
        pmh_liver_disease = int(row['pmh_liver_disease'])
        chronic_dehabilitation = int(row['chronic_dehabilitation'])
        pmh_disability_type___0 = row['pmh_disability_type___0']
        pmh_disability_type___1 = row['pmh_disability_type___1']
        pmh_disability_type___2 = row['pmh_disability_type___2']
        pmh_disability_type___3 = row['pmh_disability_type___3']
        pmh_disability_type___4 = row['pmh_disability_type___4']
        pmh_disability_type___self_catheterization = row['pmh_disability_type___self_catheterization']
        pmh_disability_type___5 = row['pmh_disability_type___5']
        pmh_major_surgery = int(row['pmh_major_surgery'])
        pmh_surgery_type = row['pmh_surgery_type']
        pmh_ckd = int(row['pmh_ckd'])
        qualifying_active_problems = row['qualifying_active_problems']
        qualifying_pmhx = row['qualifying_pmhx']
        qualifying_pshx = row['qualifying_pshx']
        qualifying_med = row['qualifying_med']
        pred_adj = row['pred_adj']
        predictors_complete = row['predictors_complete']
        ed_abx_true_qad = int(row['ed_abx_true_qad'])
        ed_abx_false_qad = int(row['ed_abx_false_qad'])
        inf_source___1 = int(row['inf_source___1'])
        inf_source___2 = int(row['inf_source___2'])
        inf_source___3 = int(row['inf_source___3'])
        inf_source___4 = int(row['inf_source___4'])
        inf_source___5 = int(row['inf_source___5'])
        inf_source___6 = int(row['inf_source___6'])
        inf_source___7 = row['inf_source___7']
        inf_source___8 = row['inf_source___8']
        infectious_source_comments = row['infectious_source_comments']
        exclusion_filter = row['exclusion_filter']
        bisc_confound = row['bisc_confound']
        exclusion_filter_comments = row['exclusion_filter_comments']
        outcomes_adj = row['outcomes_adj']
        outcomesother_descriptors_complete = row['outcomesother_descriptors_complete']

        if ed_abx_true_qad == 1 or ed_abx_false_qad == 1:  # ed_abx_true_qad if 4qad is continued and recieved abx, ed_abx_false_qad if no abx received were 4qad startedwithin 48 hrs
            qad = True
        else:
            qad = False

        if pmh_ckd == 3:
            esrd = True
        else:
            esrd = False

        if constitutional == 1 or complaint_dms == 1: #You need a way to get the model input variables that require manual chart review.  These are bacterial_symptom_complex, fatigue_ams, major_comorb, pre_ed_infection. The logic for these variables are in code lines 637-655. The input csv file is directly downloaded from redcap. Iain and others manually looked at charts and put in those variables required to get the bacterial_symptom_complex, fatigue_ams, major_comorb, pre_ed_infection variables at the end.
            fatigue_or_ams = 1 #based on constitutional or altered mental status
        else:
            fatigue_or_ams = 0

        if pmh_cancer_type___0 == 0  or pmh_cad == 1 or pmh_chf == 1 or  pmh_respiratory_illness == 1 or pmh_connective == 1 or pmh_cva == 1 or pmh_dm_both == 2 or pmh_immunocompromised  == 1 or pmh_ivdu == 1 or pmh_liver_disease != 1 or chronic_dehabilitation == 1 or pmh_major_surgery == 1 or pmh_ckd != 1:
            majorcomorbidity = 1
        else:
            majorcomorbidity= 0

        if pre_arrival_cfi == 1 or fever_pta == 1:
            pre_ed_infection = 1
        else:
            pre_ed_infection = 0

        if p_symp_complexes == 'pneumonia' or p_symp_complexes == 'abdominal' or p_symp_complexes == 'uti' or p_symp_complexes == 'msk':
            bacterialsymptomcomplex = 1
        else:
            bacterialsymptomcomplex = 0


        cad_ = True if pmh_cad == 1 else False
        chf_ = True if pmh_chf == 1 else False
        ckd_ = True if pmh_ckd >1 else False
        copd_ = True if pmh_respiratory_illness == 1 else False
        cva_ = True if pmh_cva == 1 else False
        diabetes_ = True if pmh_dm_both == 2 else False
        liver_ = True if pmh_liver_disease != 1 else False

        pneumonia_ = True if inf_source___1 == 1 else False
        uti_ = True if inf_source___2 == 1 else False
        abdominal_ = True if inf_source___3 == 1 else False
        skin_ = True if inf_source___4 == 1 else False
        other_ = True if inf_source___6 ==1 else False
        unknown_ = True if inf_source___5 ==1 else False

        print("Retrieving data for patient number {}, remaining patient number is {}".format(index, 200-index))
        patientlist.append(Patient(hospitalaccountid,encounterid, esrd, complaint_dms, fatigue_or_ams, majorcomorbidity,pre_ed_infection, bacterialsymptomcomplex, cad_, chf_, ckd_, copd_, cva_, diabetes_, liver_, pneumonia_, uti_, abdominal_, skin_, other_, unknown_))  #class gets called
        index += 1


# Redacted code excerpt

missing_predictor_list = []


for patient in patientlist:
    if patient.get_missing_predictor():
        missing_predictor_list.append(patient)
        print("missing predictor list", patient.get_hospital_account_id())
        try:
            print("gcs: ", patient.get_gcs_triage_value())
        except:
            print("no gcs")
        try:
            print("sbp: ", patient.get_sbp_triage_clipped_value())
        except:
            print("no sbp")
        try:
            print("si: ", patient.get_shock_index_triage_clipped_value())
        except:
            print("no shock index")
        try:
            print("spo2:", patient.get_spo2_triage_clipped_value())
        except:
            print("no spo2")
        try:
            print("temphigh", patient.get_temp_high_triage_clipped_value())
        except:
            print("no temphigh")
        try:
            print("templow: ", patient.get_temp_low_triage_clipped_value())
        except:
            print("no templow")
        try:
            print("resp: ", patient.get_resp_triage_clipped_value())
        except:
            print("no resp")



for patient in missing_predictor_list:
    patientlist.remove(patient)

print("there are {} remaining patients after removing those missing predictors".format(len(patientlist)))




total_rhee_count = 0
all_spo2_list = []
all_sbp_list = []
all_gcs_list = []
all_temp_high_list = []
all_temp_low_list = []
all_shock_index_list = []
all_resp_list = []
all_age_list = []


for patient in patientlist:
    total_rhee_count += patient.get_rhee_sepsis()
    all_spo2_list.append(patient.get_spo2_triage_clipped_value())
    all_sbp_list.append(patient.get_sbp_triage_clipped_value())
    all_gcs_list.append(patient.get_gcs_triage_value())
    all_temp_high_list.append(patient.get_temp_high_triage_clipped_value())
    all_temp_low_list.append(patient.get_temp_low_triage_clipped_value())
    all_shock_index_list.append(patient.get_shock_index_triage_clipped_value())
    all_resp_list.append(patient.get_resp_triage_clipped_value())
    all_age_list.append(patient.get_age())

nonseptic_count = len(patientlist) - total_rhee_count
rhee_prevalence = total_rhee_count/len(patientlist)
print("Rhee prevalence is ", rhee_prevalence)
spo2_average = sum(all_spo2_list)/len(all_spo2_list)
sbp_average = sum(all_sbp_list)/len(all_sbp_list)
gcs_average = sum(all_gcs_list)/len(all_gcs_list)
temp_high_average = sum(all_temp_high_list)/len(all_temp_high_list)
temp_low_average = sum(all_temp_low_list)/len(all_temp_low_list)
shock_index_average = sum(all_shock_index_list)/len(all_shock_index_list)
resp_average = sum(all_resp_list)/len(all_resp_list)
age_average = sum(all_age_list)/len(all_age_list)


print("spo2 average", spo2_average)
print("sbp_average", sbp_average)
print("gcs_average", gcs_average)
print("temp_high_average", temp_high_average)
print("temp_low_average", temp_low_average)
print("shock_index_average", shock_index_average)
print("resp_rate_average", resp_average)
print("age_average", age_average)

#Calculate average model score for the Full model:


fullmodel_modelscorelist = []
full_model_constant = 0

for patient in patientlist:
    # Redacted code 
full_modelscore_average = sum(fullmodel_modelscorelist) / len(fullmodel_modelscorelist)
print("full_model_constant is {} and the average probability is {}, rhee proportion is {}".format(full_model_constant, full_modelscore_average, rhee_prevalence))
if full_modelscore_average < rhee_prevalence:
    while full_modelscore_average < rhee_prevalence:
        fullmodel_modelscorelist = []
        for patient in patientlist:
            # Redacted code 
        full_modelscore_average = sum(fullmodel_modelscorelist) / len(fullmodel_modelscorelist)
        full_model_constant += 0.1
        print("full_model_constant is {} and the average probability is {}, rhee proportion is {}".format(full_model_constant, full_modelscore_average, rhee_prevalence))
elif full_modelscore_average > rhee_prevalence:
    while full_modelscore_average > rhee_prevalence:
        fullmodel_modelscorelist = []
        for patient in patientlist:
            # Redacted code 
        full_modelscore_average = sum(fullmodel_modelscorelist) / len(fullmodel_modelscorelist)
        full_model_constant -= 0.1
        print("full_model_constant is {} and the average probability is {}, rhee proportion is {}".format(full_model_constant, full_modelscore_average, rhee_prevalence))


vsmodel_modelscorelist= []
vsmodel_constant = 0

for patient in patientlist:
    vsmodel_modelscorelist.append(# Redacted code excerpt)
vs_modelscore_average = sum(vsmodel_modelscorelist)/len(vsmodel_modelscorelist)
print("vs_model_constant is {} and the average probability is {}, rhee proportion is {}".format(
            vsmodel_constant, vs_modelscore_average, rhee_prevalence))
if vs_modelscore_average < rhee_prevalence:
    while vs_modelscore_average < rhee_prevalence:
        vsmodel_modelscorelist = []
        for patient in patientlist:
            vsmodel_modelscorelist.append(# Redacted code excerpt)
        vs_modelscore_average = sum(vsmodel_modelscorelist) / len(vsmodel_modelscorelist)
        vsmodel_constant += 0.1
        print("vs_model_constant is {} and the average probability is {}, rhee proportion is {}".format(
            vsmodel_constant, vs_modelscore_average, rhee_prevalence))
elif vs_modelscore_average > rhee_prevalence:
    while vs_modelscore_average > rhee_prevalence:
        vsmodel_modelscorelist = []
        for patient in patientlist:
            vsmodel_modelscorelist.append(''' Redacted code ''')
        vs_modelscore_average = sum(vsmodel_modelscorelist) / len(vsmodel_modelscorelist)
        vsmodel_constant -= 0.1
        print("vs_model_constant is {} and the average probability is {}, rhee proportion is {}".format(
            vsmodel_constant, vs_modelscore_average, rhee_prevalence))


triage_sepsis_full_scores = []
triage_nosepsis_full_scores = []
triage_sepsis_vs_scores = []
triage_nosepsis_vs_scores = []
triage_sepsis_qsofa_scores = []
triage_nosepsis_qsofa_scores = []

for patient in patientlist:
    full_score = ''' Redacted code '''

    vs_score = ''' Redacted code '''
    patient.set_full_model_score(full_score)
    patient.set_vs_model_score(vs_score)

    if patient.get_rhee_sepsis() == 1:
        triage_sepsis_full_scores.append(full_score)
        triage_sepsis_vs_scores.append(vs_score)
        triage_sepsis_qsofa_scores.append(patient.get_qsofa())
    else:
        triage_nosepsis_full_scores.append(full_score)
        triage_nosepsis_vs_scores.append(vs_score)
        triage_nosepsis_qsofa_scores.append(patient.get_qsofa())


fullscorelist = []
vsscorelist = []
qsofalist = []
labellist = []





columns = ["hospitalaccountid", "spo2", "gcs", "predictorsbp", "age", "male_sex", "predictortemp",
           "predictorSI", "predictorresp", "fatigue_ams", "bacterialsymptomcomplex", "majorcomorbidity",
           "pre_ed_infection", "temp_low", "cultures", "qad","vasopressors","ventilator","creat_dysfunction","gfr_dysfunction","platelet_dysfucntion","tbili_dysfunction","lactate_dysfunction","sepsis", "full_modelscore", "vs_modelscore", "qsofa"]

out_file = open("rhee_predictors_scores_outcome_20211122_testing", "w")
writer = csv.DictWriter(out_file, fieldnames=columns)
writer.writeheader()



for patient in patientlist:
    writer.writerow({"hospitalaccountid": patient.get_hospital_account_id(),
                     "spo2": patient.get_spo2_triage_clipped_value(),
                     "gcs": patient.get_gcs_triage_value(),
                     "predictorsbp": patient.get_sbp_triage_clipped_value(),
                     "age": patient.get_age(),
                     "male_sex": patient.get_male_sex(),
                     "predictortemp": patient.get_temp_high_triage_clipped_value(),
                     "predictorSI": patient.get_shock_index_triage_clipped_value(),
                     "predictorresp": patient.get_resp_triage_clipped_value(),
                     "fatigue_ams": patient.get_fatigue_ams(),
                     "bacterialsymptomcomplex": patient.get_bacterial_symptom_complex(),
                     "majorcomorbidity": patient.get_major_comorbidity(),
                     "pre_ed_infection": patient.get_pre_ed_infection(),
                     "temp_low": patient.get_temp_low_triage_clipped_value(),
                     "cultures": patient.get_culture_dysfunction(),
                     "qad": patient.get_qad_dysfunction(),
                     "vasopressors": patient.get_vaso_dysfunction(),
                     "ventilator": patient.get_respiratory_dysfunction(),
                     "creat_dysfunction": patient.get_creatinine_dysfunction(),
                     "gfr_dysfunction": patient.get_gfr_dysfunction(),
                     "platelet_dysfucntion": patient.get_platelet_dysfunction(),
                     "tbili_dysfunction": patient.get_tbili_dysfunction(),
                     "lactate_dysfunction": patient.get_lactate_dysfunction(),
                     "sepsis": patient.get_rhee_sepsis(),
                     "full_modelscore": patient.get_full_model_score(),
                     "vs_modelscore": patient.get_vs_model_score(),
                     "qsofa": patient.get_qsofa(),
                     })


    fullscorelist.append(patient.get_full_model_score())
    vsscorelist.append(patient.get_vs_model_score())
    labellist.append(patient.get_rhee_sepsis())
    qsofalist.append(patient.get_qsofa())

print("fullscorelist is: ")
print(fullscorelist)
print("vsscorelist is: ")
print(vsscorelist)
print("qsofa list is: ")
print(qsofalist)
print("label list is: ")
print(labellist)



one_hr_sepsis_full_scores = []
one_hr_nosepsis_full_scores = []
one_hr_sepsis_vs_scores = []
one_hr_nosepsis_vs_scores = []
one_hr_sepsis_qsofa_scores = []
one_hr_nosepsis_qsofa_scores = []



three_hrs_sepsis_full_scores = []
three_hrs_nosepsis_full_scores = []
three_hrs_sepsis_vs_scores = []
three_hrs_nosepsis_vs_scores = []
three_hrs_sepsis_qsofa_scores = []
three_hrs_nosepsis_qsofa_scores = []


#cha
two_hrs_sepsis_full_scores = []
two_hrs_nosepsis_full_scores = []
two_hrs_sepsis_vs_scores = []
two_hrs_nosepsis_vs_scores = []
two_hrs_sepsis_qsofa_scores = []
two_hrs_nosepsis_qsofa_scores = []


four_hrs_sepsis_full_scores = []
four_hrs_nosepsis_full_scores = []
four_hrs_sepsis_vs_scores = []
four_hrs_nosepsis_vs_scores = []
four_hrs_sepsis_qsofa_scores = []
four_hrs_nosepsis_qsofa_scores = []
#end cha


six_hrs_sepsis_full_scores = []
six_hrs_nosepsis_full_scores = []
six_hrs_sepsis_vs_scores = []
six_hrs_nosepsis_vs_scores = []

#cha implementing



for patient in patientlist:

    ed_in = patient.get_edin()

    fullscorefirstflag= False
    vsscorefirstflag = False
    qsofafirstflag = False

    one_hr_full_score = None
    one_hr_vs_score = None

    # cha - sepsis score for two and four hours
    two_hrs_full_score = None
    two_hrs_vs_score = None
    two_hrs_qsofa_score = None
    four_hrs_full_score = None
    four_hrs_vs_score = None
    four_hrs_qsofa_score = None
    # cha - end of sepsis score for two and 4 hours

    three_hrs_full_score = None
    three_hrs_vs_score = None
    six_hrs_full_score = None
    six_hrs_vs_score = None


    for processedvitalset in patient.get_processed_vital_list():

        temp_high = processedvitalset.get_temp_high_cont()
        temp_low = processedvitalset.get_temp_low_cont()
        sbp_exp = processedvitalset.get_sbp_exp_cont()
        si_exp = processedvitalset.get_si_exp_cont()
        resp_exp = processedvitalset.get_resp_exp_cont()
        spo2_exp = processedvitalset.get_spo2_exp_cont()
        gcs_min = processedvitalset.get_gcs_min_cont()
        time_difference = (processedvitalset.get_time() - patient.get_edin()).total_seconds()

        gcs_qsofa = 0
        sbp_qsofa = 0
        resp_qsofa = 0


        if gcs_min < 15:
            gcs_qsofa = 1
        if sbp_exp <= 100:
            sbp_qsofa = 1
        if resp_exp >= 22:
            resp_qsofa = 1

        qsofa_ = gcs_qsofa + sbp_qsofa + resp_qsofa

        full_score = ''' Redacted code '''

        vs_score = ''' Redacted code '''


        if not qsofafirstflag:
            qsofa_max = qsofa_
            qsofafirstflag = True
        else:
            if qsofa_ > qsofa_max:
                qsofa_max = qsofa_

        if not fullscorefirstflag:
            fullscore_max = full_score
            fullscorefirstflag = True
        else:
            if full_score > fullscore_max:
                fullscore_max = full_score

        if not vsscorefirstflag:
            vsscore_max = vs_score
            vsscorefirstflag = True
        else:
            if vs_score > vsscore_max:
                vsscore_max = vs_score

        if time_difference < 3600:
            one_hr_full_score = fullscore_max
            one_hr_vs_score = vsscore_max
            one_hr_qsofa_score = qsofa_max

        if 7200 <= time_difference < 10800:
            three_hrs_full_score = fullscore_max
            three_hrs_vs_score = vsscore_max
            three_hrs_qsofa_score = qsofa_max

        # cha - sepsis score for two and four hours
        if 3600 <= time_difference < 7200:
            two_hrs_full_score = fullscore_max
            two_hrs_vs_score = vsscore_max
            two_hrs_qsofa_score = qsofa_max

        if 10800 <= time_difference < 14400:
            four_hr_vs_score = fullscore_max
            four_hrs_vs_score = vsscore_max
            four_hrs_qsofa_score = qsofa_max
        # cha - end of sepsis score for 2 and 4 hours

        if 18000 <= time_difference < 21600:
            six_hrs_full_score = fullscore_max
            six_hrs_vs_score = vsscore_max





    if one_hr_full_score is not None:
        patient.set_1hr_full_model_score(one_hr_full_score)
        patient.set_1hr_vs_model_score(one_hr_vs_score)
        patient.set_1hr_qsofa_score(one_hr_qsofa_score)
        if patient.get_rhee_sepsis()  == 1:
            one_hr_sepsis_full_scores.append(one_hr_full_score)
            one_hr_sepsis_vs_scores.append(one_hr_vs_score)
            one_hr_sepsis_qsofa_scores.append(one_hr_qsofa_score)
        elif patient.get_rhee_sepsis()  == 0:
            one_hr_nosepsis_full_scores.append(one_hr_full_score)
            one_hr_nosepsis_vs_scores.append(one_hr_vs_score)
            one_hr_nosepsis_qsofa_scores.append(one_hr_qsofa_score)

    if three_hrs_full_score is not None:
        patient.set_3hr_full_model_score(three_hrs_full_score)
        patient.set_3hr_vs_model_score(three_hrs_vs_score)
        patient.set_3hr_qsofa_score(three_hrs_qsofa_score)
        if patient.get_rhee_sepsis() == 1:
            three_hrs_sepsis_full_scores.append(three_hrs_full_score)
            three_hrs_sepsis_vs_scores.append(three_hrs_vs_score)
            three_hrs_sepsis_qsofa_scores.append(three_hrs_qsofa_score)
        elif patient.get_rhee_sepsis() == 0:
            three_hrs_nosepsis_full_scores.append(three_hrs_full_score)
            three_hrs_nosepsis_vs_scores.append(three_hrs_vs_score)
            three_hrs_nosepsis_qsofa_scores.append(three_hrs_qsofa_score)

    if six_hrs_full_score is not None:
        if patient.get_rhee_sepsis() == 1:
            six_hrs_sepsis_full_scores.append(six_hrs_full_score)
            six_hrs_sepsis_vs_scores.append(six_hrs_vs_score)
        elif patient.get_rhee_sepsis() == 0:
            six_hrs_nosepsis_full_scores.append(six_hrs_full_score)
            six_hrs_nosepsis_vs_scores.append(six_hrs_vs_score)


#cha
    if two_hrs_full_score is not None:
        patient.set_2hr_full_model_score(two_hrs_full_score)
        patient.set_2hr_vs_model_score(two_hrs_vs_score)
        patient.set_2hr_qsofa_score(two_hrs_qsofa_score)
        if patient.get_rhee_sepsis() == 1:
            two_hrs_sepsis_full_scores.append(two_hrs_full_score)
            two_hrs_sepsis_vs_scores.append(two_hrs_vs_score)
            two_hrs_sepsis_qsofa_scores.append(two_hrs_qsofa_score)
        elif patient.get_rhee_sepsis() == 0:
            two_hrs_nosepsis_full_scores.append(two_hrs_full_score)
            two_hrs_nosepsis_vs_scores.append(two_hrs_vs_score)
            two_hrs_nosepsis_qsofa_scores.append(two_hrs_qsofa_score)

    if four_hrs_full_score is not None:
        patient.set_4hr_full_model_score(four_hrs_full_score)
        patient.set_4hr_vs_model_score(four_hrs_vs_score)
        patient.set_4hr_qsofa_score(four_hrs_qsofa_score)
        if patient.get_rhee_sepsis() == 1:
            four_hrs_sepsis_full_scores.append(four_hrs_full_score)
            four_hrs_sepsis_vs_scores.append(four_hrs_vs_score)
            four_hrs_sepsis_qsofa_scores.append(four_hrs_qsofa_score)
        elif patient.get_rhee_sepsis() == 0:
            four_hrs_nosepsis_full_scores.append(four_hrs_full_score)
            four_hrs_nosepsis_vs_scores.append(four_hrs_vs_score)
            four_hrs_nosepsis_qsofa_scores.append(four_hrs_qsofa_score)


#end

    # cha - declaration for reading fullscores in 1 hr, 2hr, 3hr
    fullscorelist_1hr = []
    vsscorelist_1hr = []
    qsofalist_1hr = []
    labellist_1hr = []

    fullscorelist_2hr = []
    vsscorelist_2hr = []
    qsofalist_2hr = []
    labellist_2hr = []

    fullscorelist_3hr = []
    vsscorelist_3hr = []
    qsofalist_3hr = []
    labellist_3hr = []
    # end cha -

    columns = ["hospitalaccountid", "antibioticHr", "sepsis", "full_score_triage", "vs_score_triage", "qsofa_triage","2hrfullscore","2hrvsscore","2hrqsofascore","4hrfullscore","4hrvsscore","4hrqsofascore"]
    out_file = open(
        "cha_print_abxhr.csv", "w")
    writer = csv.DictWriter(out_file, fieldnames=columns)
    writer.writeheader()

  #  Gcss(self.hospitalaccountid, False).get_ed_triage_vital()
    for patient in patientlist:
     writer.writerow({
            "hospitalaccountid": patient.get_hospital_account_id(),
            "antibioticHr": patient.get_first_antibiotic_time(),
            "sepsis": patient.get_rhee_sepsis(),
            "full_score_triage": patient.get_full_model_score(),
            "vs_score_triage": patient.get_vs_model_score(),
            "qsofa_triage": patient.get_qsofa(),
            "2hrfullscore": patient.get_2hr_full_model_score(),
            "2hrvsscore": patient.get_2hr_vs_model_score(),
            "2hrqsofascore": patient.get_2hr_qsofa_score(),
            "4hrfullscore": patient.get_4hr_full_model_score(),
            "4hrvsscore": patient.get_4hr_vs_model_score(),
            "4hrqsofascore": patient.get_4hr_qsofa_score()
        })

    fullscorelist_2hr.append(patient.get_2hr_full_model_score())
    vsscorelist_2hr.append(patient.get_2hr_vs_model_score())
    labellist_2hr.append(patient.get_rhee_sepsis())
    qsofalist_2hr.append(patient.get_2hr_qsofa_score())

    #cha end

print("triage sepsis full scores are: ", triage_sepsis_full_scores)
print("triage nosepsis full scores are: ", triage_nosepsis_full_scores)
print("1 hr sepsis full scores are: ", one_hr_sepsis_full_scores)
print("1 hr nosepsis full scores are: ", one_hr_nosepsis_full_scores)
print("3 hrs sepsis full scores are: ", three_hrs_sepsis_full_scores)
print("3 hrs nosepsis full scores are: ", three_hrs_nosepsis_full_scores)

print("2 hrs sepsis full scores are: ", two_hrs_sepsis_full_scores)
print("2 hrs nosepsis full scores are: ", two_hrs_nosepsis_full_scores)

print("4 hrs sepsis full scores are: ", four_hrs_sepsis_full_scores)
print("4 hrs nosepsis full scores are: ", four_hrs_nosepsis_full_scores)


print("triage sepsis vs scores are: ", triage_sepsis_vs_scores)
print("triage nosepsis vs scores are: ", triage_nosepsis_vs_scores)
print("1 hr sepsis vs scores are: ", one_hr_sepsis_vs_scores)
print("1 hr nosepsis vs scores are: ", one_hr_nosepsis_vs_scores)
print("3 hrs sepsis vs scores are: ", three_hrs_sepsis_vs_scores)
print("3 hrs nosepsis vs scores are: ", three_hrs_nosepsis_vs_scores)

print("triage sepsis qsofa scores are: ", triage_sepsis_qsofa_scores)
print("triage nosepsis qsofa scores are: ", triage_nosepsis_qsofa_scores)
print("1 hr sepsis qsofa scores are: ", one_hr_sepsis_qsofa_scores)
print("1 hr nosepsis qsofa scores are: ", one_hr_nosepsis_qsofa_scores)
print("3 hrs sepsis qsofa scores are: ", three_hrs_sepsis_qsofa_scores)
print("3 hrs nosepsis qsofa scores are: ", three_hrs_nosepsis_qsofa_scores)

sepsis_no_abx_count = 0
nosepsis_no_abx_count = 0
sepsis_abx_no_hypotension_count = 0
nosepsis_abx_no_hypotension_count = 0
sepsis_abx_less_than_1_hr_of_hypotension_count = 0
nosepsis_abx_less_than_1_hr_of_hypotension_count = 0
sepsis_abx_1_to_3_hr_of_hypotension_count = 0
nosepsis_abx_1_to_3_hr_of_hypotension_count = 0
sepsis_abx_more_than_3_hr_of_hypotension_count = 0
nosepsis_abx_more_than_3_hr_of_hypotension_count = 0



sepsis_wbc_list = []
nosepsis_wbc_list = []


sepsis_no_wbc_count = 0
nosepsis_no_wbc_count = 0
sepsis_wbc_lt_1_hr_of_edin_count = 0
nosepsis_wbc_lt_1_hr_of_edin_count = 0
sepsis_wbc_1_to_3_hr_of_edin_count = 0
nosepsis_wbc_1_to_3_hr_of_edin_count = 0
sepsis_wbc_gt_3_hrs_of_edin_count = 0
nosepsis_wbc_gt_3_hrs_of_edin_count = 0



sepsis_lactate_list = []
nosepsis_lactate_list = []


sepsis_no_lactate_count = 0
nosepsis_no_lactate_count = 0
sepsis_lactate_lt_1_hr_of_edin_count = 0
nosepsis_lactate_lt_1_hr_of_edin_count = 0
sepsis_lactate_1_to_3_hr_of_edin_count = 0
nosepsis_lactate_1_to_3_hr_of_edin_count = 0
sepsis_lactate_gt_3_hrs_of_edin_count = 0
nosepsis_lactate_gt_3_hrs_of_edin_count = 0




septic_hypotensive_count = 0
nonseptic_hypotensive_count = 0

septic_male_count = 0
nonseptic_male_count = 0

septic_nonwhite_count = 0
nonseptic_nonwhite_count = 0

septic_mortality_count = 0
nonseptic_mortality_count = 0

septic_missing_sofa_count = 0
nonseptic_missing_sofa_count = 0




septic_pneumonia_count = 0
septic_uti_count = 0
septic_abdominal_infection_count = 0
septic_skin_soft_tissue_infection_count = 0
septic_other_infection_count = 0
septic_unknown_infection_count = 0

septic_cad_count = 0
septic_chf_count = 0
septic_ckd_count = 0
septic_copd_count = 0
septic_cva_count = 0
septic_diabetes_count = 0
septic_liver_disease_count = 0

nonseptic_cad_count = 0
nonseptic_chf_count = 0
nonseptic_ckd_count = 0
nonseptic_copd_count = 0
nonseptic_cva_count = 0
nonseptic_diabetes_count = 0
nonseptic_liver_disease_count = 0





septic_age_list = []
nonseptic_age_list = []
septic_triage_sbp_list = []
nonseptic_triage_sbp_list = []
septic_triage_hr_list = []
nonseptic_triage_hr_list = []
septic_triage_gcs_list = []
nonseptic_triage_gcs_list = []
septic_triage_resp_list = []
nonseptic_triage_resp_list = []
septic_triage_spo2_list = []
nonseptic_triage_spo2_list = []
septic_triage_temp_list = []
nonseptic_triage_temp_list = []
septic_first_lactate_list = []
nonseptic_first_lactate_list = []
septic_first_creatinine_list = []
nonseptic_first_creatinine_list = []
septic_first_platelet_list = []
nonseptic_first_platelet_list = []
septic_first_wbc_list = []
nonseptic_first_wbc_list = []
septic_sofa_list = []
nonseptic_sofa_list = []


sepsis_time_to_hypotension_list = []
nosepsis_time_to_hypotension_list = []


septic_encounter_id_list = []
nonseptic_encounter_id_list = []


delayed_antibiotics_patient_list = []
#antibiotic_given_hr_list =[]   #cha

for patient in patientlist:
    print(patient.get_hospital_account_id())
    first_hypotension_flag = False
    edin_ = patient.get_edin()
    sepsis_ = patient.get_rhee_sepsis()
    first_wbc_ = patient.get_first_wbc()
    first_lactate_ = patient.get_first_lactate()
    sofa_ = patient.get_sofa()
   # antibiotic_given_hr_list.append(patient.get_first_antibiotic_time()) #cha

    if sepsis_ == 1:
        septic_encounter_id_list.append(patient.get_patient_encounter_id())

        septic_age_list.append(patient.get_age())
        septic_triage_sbp_list.append(patient.get_sbp_triage_unclipped_value())
        septic_triage_hr_list.append(patient.get_hr_triage_unclipped_value())
        septic_triage_gcs_list.append(patient.get_gcs_triage_value())
        septic_triage_resp_list.append(patient.get_resp_triage_unclipped_value())
        septic_triage_spo2_list.append(patient.get_spo2_triage_unclipped_value())
        septic_triage_temp_list.append(patient.get_temp_triage_unclipped_value())
        if patient.get_first_creatinine() is not None:
            septic_first_creatinine_list.append(patient.get_first_creatinine().get_value())
        if patient.get_first_platelet() is not None:
            septic_first_platelet_list.append(patient.get_first_platelet().get_value())

        if patient.get_male_sex() == 1:
            septic_male_count += 1
        if patient.get_white_race() == 0:
            septic_nonwhite_count += 1
        if patient.get_mortality():
            septic_mortality_count += 1
        if not isinstance(patient.get_first_antibiotic_time(), datetime.datetime):
            sepsis_no_abx_count += 1
        if first_wbc_ is None:
            sepsis_no_wbc_count += 1
        if first_lactate_ is None:
            sepsis_no_lactate_count += 1
        if sofa_ is None:
            septic_missing_sofa_count += 1
        else:
            septic_sofa_list.append(sofa_)

        if patient.get_pneumonia():
            septic_pneumonia_count += 1
        if patient.get_uti():
            septic_uti_count += 1
        if patient.get_abdominal():
            septic_abdominal_infection_count += 1
        if patient.get_skin():
            septic_skin_soft_tissue_infection_count += 1
        if patient.get_other():
            septic_other_infection_count += 1
        if patient.get_unknown():
            septic_unknown_infection_count += 1

        if patient.get_cad():
            septic_cad_count += 1
        if patient.get_chf():
            septic_chf_count += 1
        if patient.get_ckd():
            septic_ckd_count += 1
        if patient.get_copd():
            septic_copd_count += 1
        if patient.get_cva():
            septic_cva_count += 1
        if patient.get_diabetes():
            septic_diabetes_count += 1
        if patient.get_liver_disease():
            septic_liver_disease_count += 1

    else:
        nonseptic_encounter_id_list.append(patient.get_patient_encounter_id())

        nonseptic_age_list.append(patient.get_age())
        nonseptic_triage_sbp_list.append(patient.get_sbp_triage_unclipped_value())
        nonseptic_triage_hr_list.append(patient.get_hr_triage_unclipped_value())
        nonseptic_triage_gcs_list.append(patient.get_gcs_triage_value())
        nonseptic_triage_resp_list.append(patient.get_resp_triage_unclipped_value())
        nonseptic_triage_spo2_list.append(patient.get_spo2_triage_unclipped_value())
        nonseptic_triage_temp_list.append(patient.get_temp_triage_unclipped_value())
        if patient.get_first_creatinine() is not None:
            nonseptic_first_creatinine_list.append(patient.get_first_creatinine().get_value())
        if patient.get_first_platelet() is not None:
            nonseptic_first_platelet_list.append(patient.get_first_platelet().get_value())

        if patient.get_male_sex() == 1:
            nonseptic_male_count += 1
        if patient.get_white_race() == 0:
            nonseptic_nonwhite_count += 1
        if patient.get_mortality():
            nonseptic_mortality_count += 1
        if not isinstance(patient.get_first_antibiotic_time(), datetime.datetime):
            nosepsis_no_abx_count += 1
        if first_wbc_ is None:
            nosepsis_no_wbc_count += 1
        if first_lactate_ is None:
            nosepsis_no_lactate_count += 1
        if sofa_ is None:
            nonseptic_missing_sofa_count += 1
        else:
            nonseptic_sofa_list.append(sofa_)

        if patient.get_cad():
            nonseptic_cad_count += 1
        if patient.get_chf():
            nonseptic_chf_count += 1
        if patient.get_ckd():
            nonseptic_ckd_count += 1
        if patient.get_copd():
            nonseptic_copd_count += 1
        if patient.get_cva():
            nonseptic_cva_count += 1
        if patient.get_diabetes():
            nonseptic_diabetes_count += 1
        if patient.get_liver_disease():
            nonseptic_liver_disease_count += 1


    for vitalset in patient.get_unprocessed_vital_list():
        if vitalset.get_sbp_cont() is not None:
            if vitalset.get_sbp_cont().get_value() < 90 and not first_hypotension_flag:
                accountid_ = patient.get_hospital_account_id()
                sbp_ = vitalset.get_sbp_cont().get_value()
                timefromtriage_ = (vitalset.get_sbp_cont().get_time() - edin_).total_seconds()/60  #time in minutes
                first_hypotention_time_ = vitalset.get_sbp_cont().get_time()
                if sepsis_ == 1:
                    sepsis_time_to_hypotension_list.append(timefromtriage_)
                    septic_hypotensive_count += 1
                else:
                    nosepsis_time_to_hypotension_list.append(timefromtriage_)
                    nonseptic_hypotensive_count += 1

                if isinstance(patient.get_first_antibiotic_time(), datetime.datetime):
                    first_abx_time_ = patient.get_first_antibiotic_time()
                    abx_hypotention_time_diff_ = (first_abx_time_ - first_hypotention_time_).total_seconds()

                    if abx_hypotention_time_diff_ < 3600:
                        if sepsis_ == 1:
                            sepsis_abx_less_than_1_hr_of_hypotension_count += 1
                        else:
                            nosepsis_abx_less_than_1_hr_of_hypotension_count += 1
                    elif 3600 <= abx_hypotention_time_diff_ < 10800:
                        if sepsis_ == 1:
                            sepsis_abx_1_to_3_hr_of_hypotension_count += 1
                        else:
                            nosepsis_abx_1_to_3_hr_of_hypotension_count += 1
                    else:
                        if sepsis_ == 1:
                            sepsis_abx_more_than_3_hr_of_hypotension_count += 1
                            delayed_antibiotics_patient_list.append(patient.get_hospital_account_id())
                        else:
                            nosepsis_abx_more_than_3_hr_of_hypotension_count += 1
                else:
                    first_abx_time_ = 'no abx'
                    abx_hypotention_time_diff_ = 'no abx'

                #print("hospitalaccountid is {} , sbp at first hypotension is  {}, time from triage is {}, hypotention time is {}, first abx time is {}, abx_hypotension time diff is {}, and the patient is rhee_sepsis {}".format(accountid_,sbp_,timefromtriage_,first_hypotention_time_, first_abx_time_, abx_hypotention_time_diff_, sepsis_))
                first_hypotension_flag = True

    if not first_hypotension_flag and isinstance(patient.get_first_antibiotic_time(), datetime.datetime):
        if sepsis_ == 1:
            sepsis_abx_no_hypotension_count += 1
        if sepsis_ == 0:
            nosepsis_abx_no_hypotension_count += 1



    if first_wbc_ is not None:
        wbc_time_from_ed_in_ = (first_wbc_.get_result_time() - edin_).total_seconds()
        if sepsis_ == 1:
            septic_first_wbc_list.append(first_wbc_.get_value())
            if wbc_time_from_ed_in_ <3600:
                sepsis_wbc_lt_1_hr_of_edin_count += 1
            elif 3600 <= wbc_time_from_ed_in_ < 10800:
                sepsis_wbc_1_to_3_hr_of_edin_count += 1
            else:
                sepsis_wbc_gt_3_hrs_of_edin_count  += 1
        else:
            nonseptic_first_wbc_list.append(first_wbc_.get_value())
            if wbc_time_from_ed_in_ <3600:
                nosepsis_wbc_lt_1_hr_of_edin_count += 1
            elif 3600 <= wbc_time_from_ed_in_ < 10800:
                nosepsis_wbc_1_to_3_hr_of_edin_count += 1
            else:
                nosepsis_wbc_gt_3_hrs_of_edin_count += 1

    if first_lactate_ is not None:
        lact_time_from_ed_in_ = (first_lactate_.get_result_time() - edin_).total_seconds()
        if sepsis_ == 1:
            septic_first_lactate_list.append(first_lactate_.get_value())
            if lact_time_from_ed_in_ <3600:
                sepsis_lactate_lt_1_hr_of_edin_count += 1
            elif 3600 <= lact_time_from_ed_in_ < 10800:
                sepsis_lactate_1_to_3_hr_of_edin_count += 1
            else:
                sepsis_lactate_gt_3_hrs_of_edin_count  += 1
        else:
            nonseptic_first_lactate_list.append(first_lactate_.get_value())
            if lact_time_from_ed_in_ <3600:
                nosepsis_lactate_lt_1_hr_of_edin_count += 1
            elif 3600 <= lact_time_from_ed_in_ < 10800:
                nosepsis_lactate_1_to_3_hr_of_edin_count += 1
            else:
                nosepsis_lactate_gt_3_hrs_of_edin_count += 1









septic_age = median_25th_75th_percentile(septic_age_list)
septic_sbp = median_25th_75th_percentile(septic_triage_sbp_list)
septic_time_to_hypotension = median_25th_75th_percentile(sepsis_time_to_hypotension_list)
septic_hr = median_25th_75th_percentile(septic_triage_hr_list)
septic_gcs = median_25th_75th_percentile(septic_triage_gcs_list)
septic_resp = median_25th_75th_percentile(septic_triage_resp_list)
septic_spo2 = median_25th_75th_percentile(septic_triage_spo2_list)
septic_temp = median_25th_75th_percentile(septic_triage_temp_list)
septic_lactate = median_25th_75th_percentile(septic_first_lactate_list)
septic_creatinine = median_25th_75th_percentile(septic_first_creatinine_list)
septic_platelet = median_25th_75th_percentile(septic_first_platelet_list)
septic_wbc = median_25th_75th_percentile(septic_first_wbc_list)
septic_sofa = median_25th_75th_percentile(septic_sofa_list)
septic_vsscoreCI = median_25th_75th_percentile(vsscorelist)

nonseptic_age = median_25th_75th_percentile(nonseptic_age_list)
nonseptic_sbp = median_25th_75th_percentile(nonseptic_triage_sbp_list)
nonseptic_time_to_hypotension = median_25th_75th_percentile(nosepsis_time_to_hypotension_list)
nonseptic_hr = median_25th_75th_percentile(nonseptic_triage_hr_list)
nonseptic_gcs = median_25th_75th_percentile(nonseptic_triage_gcs_list)
nonseptic_resp = median_25th_75th_percentile(nonseptic_triage_resp_list)
nonseptic_spo2 = median_25th_75th_percentile(nonseptic_triage_spo2_list)
nonseptic_temp = median_25th_75th_percentile(nonseptic_triage_temp_list)
nonseptic_lactate = median_25th_75th_percentile(nonseptic_first_lactate_list)
nonseptic_creatinine = median_25th_75th_percentile(nonseptic_first_creatinine_list)
nonseptic_platelet = median_25th_75th_percentile(nonseptic_first_platelet_list)
nonseptic_wbc = median_25th_75th_percentile(nonseptic_first_wbc_list)
nonseptic_sofa = median_25th_75th_percentile(nonseptic_sofa_list)
nonseptic_vsscoreCI = median_25th_75th_percentile(fullmodel_modelscorelist)


fields_1 = ['Field_Name', 'Sepsis_fraction', 'Non_sepsis_fraction', 'Sepsis_count', 'Non_sepsis_count' ]

sepsis_hypotention_abx_total = sepsis_abx_less_than_1_hr_of_hypotension_count + sepsis_abx_1_to_3_hr_of_hypotension_count + sepsis_abx_more_than_3_hr_of_hypotension_count
nosepsis_hypotension_abx_total = nosepsis_abx_less_than_1_hr_of_hypotension_count + nosepsis_abx_1_to_3_hr_of_hypotension_count + nosepsis_abx_more_than_3_hr_of_hypotension_count

sepsis_wbc_total = sepsis_wbc_lt_1_hr_of_edin_count + sepsis_wbc_1_to_3_hr_of_edin_count + sepsis_wbc_gt_3_hrs_of_edin_count
nosepsis_wbc_total = nosepsis_wbc_lt_1_hr_of_edin_count + nosepsis_wbc_1_to_3_hr_of_edin_count + nosepsis_wbc_gt_3_hrs_of_edin_count

sepsis_lactate_total = sepsis_lactate_lt_1_hr_of_edin_count + sepsis_lactate_1_to_3_hr_of_edin_count + sepsis_lactate_gt_3_hrs_of_edin_count
nosepsis_lactate_total = nosepsis_lactate_lt_1_hr_of_edin_count + nosepsis_lactate_1_to_3_hr_of_edin_count + nosepsis_lactate_gt_3_hrs_of_edin_count


demographics = []

demographics.append(['Non-White', septic_nonwhite_count/total_rhee_count, nonseptic_nonwhite_count/nonseptic_count, septic_nonwhite_count, nonseptic_nonwhite_count])
demographics.append(['Male', septic_male_count/total_rhee_count, nonseptic_male_count/nonseptic_count, septic_male_count, nonseptic_male_count])
demographics.append(['ED_hypotension', septic_hypotensive_count/total_rhee_count, nonseptic_hypotensive_count/nonseptic_count,septic_hypotensive_count,nonseptic_hypotensive_count])
demographics.append(['Mortality', septic_mortality_count/total_rhee_count, nonseptic_mortality_count/nonseptic_count,septic_mortality_count, nonseptic_mortality_count])
demographics.append(['no_abx',sepsis_no_abx_count/total_rhee_count, nosepsis_no_abx_count/nonseptic_count, sepsis_no_abx_count, nosepsis_no_abx_count])
demographics.append(['abx_no_hypotension', sepsis_abx_no_hypotension_count/total_rhee_count, nosepsis_abx_no_hypotension_count/nonseptic_count,sepsis_abx_no_hypotension_count, nosepsis_abx_no_hypotension_count])
demographics.append(['abx_less_than_1_hr_of_hypotension', sepsis_abx_less_than_1_hr_of_hypotension_count/total_rhee_count, nosepsis_abx_less_than_1_hr_of_hypotension_count/nonseptic_count,sepsis_abx_less_than_1_hr_of_hypotension_count,nosepsis_abx_less_than_1_hr_of_hypotension_count])
demographics.append(['abx 1 to 3hr of hypotension', sepsis_abx_1_to_3_hr_of_hypotension_count/total_rhee_count, nosepsis_abx_1_to_3_hr_of_hypotension_count/nonseptic_count, sepsis_abx_1_to_3_hr_of_hypotension_count, nosepsis_abx_1_to_3_hr_of_hypotension_count])
demographics.append(['abx_more_than_3_hr_of_hypotension', sepsis_abx_more_than_3_hr_of_hypotension_count/total_rhee_count, nosepsis_abx_more_than_3_hr_of_hypotension_count/nonseptic_count, sepsis_abx_more_than_3_hr_of_hypotension_count,nosepsis_abx_more_than_3_hr_of_hypotension_count ])
demographics.append(['no_wbc', sepsis_no_wbc_count/total_rhee_count, nosepsis_no_wbc_count/nonseptic_count, sepsis_no_wbc_count, nosepsis_no_wbc_count])
demographics.append(['wbc_lt_1_hr_of_edin_', sepsis_wbc_lt_1_hr_of_edin_count/total_rhee_count, nosepsis_wbc_lt_1_hr_of_edin_count/nonseptic_count,sepsis_wbc_lt_1_hr_of_edin_count,nosepsis_wbc_lt_1_hr_of_edin_count])
demographics.append(['wbc_1_to_3_hr_of_edin_', sepsis_wbc_1_to_3_hr_of_edin_count/total_rhee_count, nosepsis_wbc_1_to_3_hr_of_edin_count/nonseptic_count,sepsis_wbc_1_to_3_hr_of_edin_count,nosepsis_wbc_1_to_3_hr_of_edin_count])
demographics.append(['wbc_gt_3_hrs_of_edin_', sepsis_wbc_gt_3_hrs_of_edin_count/total_rhee_count, nosepsis_wbc_gt_3_hrs_of_edin_count/nonseptic_count,sepsis_wbc_gt_3_hrs_of_edin_count,nosepsis_wbc_gt_3_hrs_of_edin_count])
demographics.append(['no_lactate', sepsis_no_lactate_count/total_rhee_count, nosepsis_no_lactate_count/nonseptic_count, sepsis_no_lactate_count, nosepsis_no_lactate_count])
demographics.append(['lactate_lt_1_hr_of_edin_', sepsis_lactate_lt_1_hr_of_edin_count/total_rhee_count, nosepsis_lactate_lt_1_hr_of_edin_count/nonseptic_count, sepsis_lactate_lt_1_hr_of_edin_count,nosepsis_lactate_lt_1_hr_of_edin_count])
demographics.append(['lactate_1_to_3_hr_of_edin_', sepsis_lactate_1_to_3_hr_of_edin_count/total_rhee_count, nosepsis_lactate_1_to_3_hr_of_edin_count/nonseptic_count,sepsis_lactate_1_to_3_hr_of_edin_count,nosepsis_lactate_1_to_3_hr_of_edin_count])
demographics.append(['lactate_gt_3_hrs_of_edin_', sepsis_lactate_gt_3_hrs_of_edin_count/total_rhee_count, nosepsis_lactate_gt_3_hrs_of_edin_count/nonseptic_count,sepsis_lactate_gt_3_hrs_of_edin_count,nosepsis_lactate_gt_3_hrs_of_edin_count])
demographics.append(['missing_sofa', septic_missing_sofa_count/total_rhee_count, nonseptic_missing_sofa_count/nonseptic_count, septic_missing_sofa_count, nonseptic_missing_sofa_count])
demographics.append(['cad', septic_cad_count/total_rhee_count, nonseptic_cad_count/nonseptic_count, septic_cad_count, nonseptic_cad_count])
demographics.append(['chf', septic_chf_count/total_rhee_count, nonseptic_chf_count/nonseptic_count, septic_chf_count, nonseptic_chf_count])
demographics.append(['ckd', septic_ckd_count/total_rhee_count, nonseptic_ckd_count/nonseptic_count, septic_ckd_count, nonseptic_ckd_count])
demographics.append(['copd', septic_copd_count/total_rhee_count, nonseptic_copd_count/nonseptic_count, septic_copd_count, nonseptic_copd_count])
demographics.append(['cva', septic_cva_count/total_rhee_count, nonseptic_cva_count/nonseptic_count, septic_cva_count, nonseptic_cva_count])
demographics.append(['diabetes', septic_diabetes_count/total_rhee_count, nonseptic_diabetes_count/nonseptic_count, septic_diabetes_count, nonseptic_diabetes_count])
demographics.append(['liver_disease', septic_liver_disease_count/total_rhee_count, nonseptic_liver_disease_count/nonseptic_count, septic_liver_disease_count, nonseptic_liver_disease_count])
demographics.append(['pneumonia', septic_pneumonia_count/total_rhee_count, 'n/a', septic_pneumonia_count, 'n/a'])
demographics.append(['uti', septic_uti_count/total_rhee_count, 'n/a', septic_uti_count, 'n/a'])
demographics.append(['abdominal', septic_abdominal_infection_count/total_rhee_count, 'n/a', septic_abdominal_infection_count, 'n/a'])
demographics.append(['skin', septic_skin_soft_tissue_infection_count/total_rhee_count, 'n/a', septic_skin_soft_tissue_infection_count, 'n/a'])
demographics.append(['other', septic_other_infection_count/total_rhee_count, 'n/a', septic_other_infection_count, 'n/a'])
demographics.append(['unknown', septic_unknown_infection_count/total_rhee_count, 'n/a', septic_unknown_infection_count, 'n/a'])


fields_2 = ['field name', 'sepsis (25th,75th %tile)', 'non-sepsis (25th, 75th %tile)']

demographics_2 = []
demographics_2.append(['age', septic_age, nonseptic_age])
demographics_2.append(['triage_sbp', septic_sbp, nonseptic_sbp])
demographics_2.append(['time_to_hypotension', septic_time_to_hypotension, nonseptic_time_to_hypotension])
demographics_2.append(['triage_hr', septic_hr, nonseptic_hr])
demographics_2.append(['triage_gcs', septic_gcs, nonseptic_gcs])
demographics_2.append(['triage_resp', septic_resp, nonseptic_resp])
demographics_2.append(['triage_spo2', septic_spo2, nonseptic_spo2])
demographics_2.append(['triage_temp', septic_temp, nonseptic_temp])
demographics_2.append(['first_lactate', septic_lactate, nonseptic_lactate])
demographics_2.append(['first_creatinine', septic_creatinine, nonseptic_creatinine])
demographics_2.append(['first_platelet', septic_platelet, nonseptic_platelet])
demographics_2.append(['first_wbc', septic_wbc, nonseptic_wbc])
demographics_2.append(['SOFA', septic_sofa, nonseptic_sofa])
demographics_2.append(['testingCI', septic_vsscoreCI, nonseptic_vsscoreCI])

with open('demographics1_20211122_testing.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fields_1)
    writer.writerows(demographics)


with open('demographics2_20211122_testing.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fields_2)
    writer.writerows(demographics_2)

print("septic encounterid list")
print(septic_encounter_id_list)
print('nonseptic encounterid list')
print(nonseptic_encounter_id_list)




print("delayed_antibiotics_septic_patient_list")
print(delayed_antibiotics_patient_list)
# print("sepsis_abx_less_than_1_hr_of_hypotension_count {}".format(sepsis_abx_less_than_1_hr_of_hypotension_count))
# print("nosepsis_abx_less_than_1_hr_of_hypotension_count {}".format(nosepsis_abx_less_than_1_hr_of_hypotension_count))
# print("sepsis_abx 1 to 3hr of hypotension count {}".format(sepsis_abx_1_to_3_hr_of_hypotension_count))
# print("nosepsis_abx 1 to 3hr of hypotension count {}".format(nosepsis_abx_1_to_3_hr_of_hypotension_count))
# print("sepsis_abx_more_than_3_hr_of_hypotension_count {}".format(sepsis_abx_more_than_3_hr_of_hypotension_count))
# print("nosepsis_abx_more_than_3_hr_of_hypotension_count {}".format(nosepsis_abx_more_than_3_hr_of_hypotension_count))
# print()
# print("sepsis_wbc_lt_1_hr_of_edin_count count {}".format(sepsis_wbc_lt_1_hr_of_edin_count))
# print("nosepsis_wbc_lt_1_hr_of_edin_count count {}".format(nosepsis_wbc_lt_1_hr_of_edin_count))
# print("sepsis_wbc_1_to_3_hr_of_edin_count count {}".format(sepsis_wbc_1_to_3_hr_of_edin_count))
# print("nosepsis_wbc_1_to_3_hr_of_edin_count count {}".format(nosepsis_wbc_1_to_3_hr_of_edin_count))
# print("sepsis_wbc_gt_3_hrs_of_edin_count count {}".format(sepsis_wbc_gt_3_hrs_of_edin_count))
# print("nosepsis_wbc_gt_3_hrs_of_edin_count count {}".format(nosepsis_wbc_gt_3_hrs_of_edin_count))
# print()
# print("sepsis_lactate_lt_1_hr_of_edin_count count {}".format(sepsis_lactate_lt_1_hr_of_edin_count))
# print("nosepsis_lactate_lt_1_hr_of_edin_count count {}".format(nosepsis_lactate_lt_1_hr_of_edin_count))
# print("sepsis_lactate_1_to_3_hr_of_edin_count count {}".format(sepsis_lactate_1_to_3_hr_of_edin_count))
# print("nosepsis_lactate_1_to_3_hr_of_edin_count count {}".format(nosepsis_lactate_1_to_3_hr_of_edin_count))
# print("sepsis_lactate_gt_3_hrs_of_edin_count count {}".format(sepsis_lactate_gt_3_hrs_of_edin_count))
# print("nosepsis_lactate_gt_3_hrs_of_edin_count count {}".format(nosepsis_lactate_gt_3_hrs_of_edin_count))
#

columns = ["hospitalaccountid", "encounterid", "ed_in", "sepsis", "full_score_triage", "vs_score_triage", "qsofa_triage", "1hrfullmax", "1hrvsmax", "1hrqsofamax", "3hrfullmax", "3hrvsmax", "3hrqsofamax"]
out_file = open("redcap_200_patients_triage_1hr_3hr_scores_20211122_testing.csv", "w")
writer = csv.DictWriter(out_file, fieldnames=columns)
writer.writeheader()



for patient in patientlist:

        writer.writerow({
            "hospitalaccountid": patient.get_hospital_account_id(),
            "encounterid": patient.get_patient_encounter_id(),
            "ed_in": patient.get_edin(),
            "sepsis": patient.get_rhee_sepsis(),
            "full_score_triage": patient.get_full_model_score(),
            "vs_score_triage": patient.get_vs_model_score(),
            "qsofa_triage": patient.get_qsofa(),
            "1hrfullmax": patient.get_1hr_full_model_score(),
            "1hrvsmax": patient.get_1hr_vs_model_score(),
            "1hrqsofamax": patient.get_1hr_qsofa_score(),
            "3hrfullmax": patient.get_3hr_full_model_score(),
            "3hrvsmax":  patient.get_3hr_vs_model_score(),
            "3hrqsofamax": patient.get_3hr_qsofa_score()
        })
        fullscorelist_1hr.append(patient.get_1hr_full_model_score())
        vsscorelist_1hr.append(patient.get_1hr_vs_model_score())
        labellist_1hr.append(patient.get_rhee_sepsis())
        qsofalist_1hr.append(patient.get_1hr_qsofa_score())

        fullscorelist_3hr.append(patient.get_3hr_full_model_score())
        vsscorelist_3hr.append(patient.get_3hr_vs_model_score())
        labellist_3hr.append(patient.get_rhee_sepsis())
        qsofalist_3hr.append(patient.get_3hr_qsofa_score())
print("fullscorelist_1hr")
print(fullscorelist_1hr)
print('vsscorelist_1hr')
print(labellist_1hr)
print('labellist_1hr')
print(labellist_1hr)
print("fullscorelist_3hr")
print(fullscorelist_3hr)
print('vsscorelist_3hr')
print(vsscorelist_3hr)
print('labellist_3hr')
print(labellist_3hr)
# missingGCSList =[]
# for patient in patientlist:
#         missingGCSList.append(Gcss(patient.get_hospital_account_id(), False).get_ed_triage_vital())
#         missingGCSList.append(patient.get_hospital_account_id())
# print("missingGCSList", missingGCSList)

full_spo2_or = 0 #per 5%
full_sbp_or = 0 #per 10mmHg
full_gcs_or = 0
full_temp_high_or = 0
full_age_or = 0 #per 10 years
full_male_sex_or = 0
full_bacterial_symptom_complex_or = 0
full_shock_index_or = 0 #per 0.25mmhg/bpm
full_resp_or = 0 #per 5 min^-1
full_fatigue_ams_or = 0
full_major_comorb_or = 0
full_pre_ed_infection_or = 0


full_score = ''' Redacted code '''

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot


vs_auc = roc_auc_score(labellist, vsscorelist)
fullmodel_auc = roc_auc_score(labellist, fullscorelist)
qsofa_auc = roc_auc_score(labellist, qsofalist)

print("roc_auc for VS model",vs_auc)
print("roc_auc for fullmodel",fullmodel_auc)
print("roc_auc for qsofa",qsofa_auc)
