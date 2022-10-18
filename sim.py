import pandas as pd
import numpy as np
from japandata.furusatonouzei.data import furusato_df as fdf
from japandata.chihoukoufuzei.data import local_df as cdf
from japandata.chihoukoufuzei.data import pref_df as cdf_pref

import matplotlib.pyplot as plt

from samplot.baseplot import BasePlot
from samplot.circusboy import CircusBoy

# Next steps
# Look at the is a payback effect from the debt in the following year.
# Look at how the subsidy money is distributed -- actually put into the same futsuu money pool?

# Constraints:
# 1 - (Before implementing the seppan funding): Total ckz should be the same
# 2 - If the total FN amount is zero, total debt amount (and everything else) should be the same
# 3 - (Before implementing the debt payback): Only touch the special-debt, final-demand, ckz
# 4 - final-demand + special-debt = demand-pre-debt
# 5 - final-demand - income should be nearly equal to ckz (up the small adjustment factor and ignoring fukoufudantai)
# 6 - The formula for debt is nominally constant * (yeardf['demand-pre-debt']-yeardf['income']) * f(economic-strength-index-prev3yearavg) with the constant chosen to that the total bond amount leads to a ckz of the desired size. I don't know the functional form of the f (and it changes every year) but it doesn't matter since I can just reuse the same debt-scaling-factor before and after removing FN (and change the constant)
## There is an exception to how the CKZ for tokyo is computed: basically it can never get ckz

# seppan data from the soumushou graph
chou = 10**12
shortfall = {
    2022: 0 * chou,
    2021: 1.7 * 2 * chou,
    2020: 0 * chou,
    2019: 0 * chou,
    2018: 0.2 * 2 * chou,
    2017: 0.7 * 2 * chou,
    2016: 0.3 * 2 * chou,
    2015: 1.5 * 2 * chou,
}


def undo_fn_sim(year_df):
    """Take all the data which includes FN and add -noFN cols"""

    simYear_df = year_df.copy()
    simYear_df["ckz-preAdj"] = (
        simYear_df["final-demand"] - simYear_df["income"]
    )  # Amount of ckz before final adjustment
    simYear_df.loc[
        simYear_df["ckz-preAdj"] < 0, "ckz-preAdj"
    ] = 0  # ckz cant be negative

    # For places without a 3-year ESI, use current year ESI
    simYear_df.loc[
        simYear_df["economic-strength-index-prev3yearavg"].isna(),
        "economic-strength-index-prev3yearavg",
    ] = simYear_df["economic-strength-index"]
    assert np.sum(simYear_df["economic-strength-index-prev3yearavg"].isna()) == 0

    # The debt scaling constant that was used for each city in practice
    # simYear_df['debt-scaling-factor'] = simYear_df['special-debt'] / \
    #     (simYear_df['demand-pre-debt']-simYear_df['income']) / \
    #     simYear_df['economic-strength-index-prev3yearavg']
    simYear_df["debt-scaling-factor"] = simYear_df["special-debt"] / (
        simYear_df["demand-pre-debt"] - simYear_df["income"]
    )
    # The adjustment factor that was used to reduce the final demand and get the ckz
    simYear_df["adjustment-factor"] = (
        1 - (simYear_df["ckz"] + simYear_df["income"]) / simYear_df["final-demand"]
    )
    simYear_df.loc[simYear_df["ckz"] == 0, "adjustment-factor"] = 0

    # Now to undo the FN effect by adding back 0.75*deductions
    simYear_df["income-noFN"] = simYear_df["income"] + 0.75 * simYear_df["deductions"]
    # With this higher income, recompute how much debt they would have been allowed to issue

    if shortfall[chihoukoufuzeiyear] > 0:
        subsidyFraction = 0.5
    else:
        subsidyFraction = 0
    print("Using subsidy fraction", subsidyFraction)
    governmentSubsidyTotal = (
        subsidyFraction * 0.75 * simYear_df["deductions"].sum()
    )  # For testing purposes

    # Find a uniform scaling of the debt-scaling-factors such that with the nouzei undone we still get the right total ckz-preAdj
    from scipy.optimize import minimize

    def ckzPreAdjConstraint(extraFactor):
        # simYear_df['special-debt-noFN'] = (extraFactor*simYear_df['debt-scaling-factor']*(
        #     simYear_df['demand-pre-debt']-simYear_df['income-noFN'])*simYear_df['economic-strength-index-prev3yearavg'])
        simYear_df["special-debt-noFN"] = (
            extraFactor
            * simYear_df["debt-scaling-factor"]
            * (simYear_df["demand-pre-debt"] - simYear_df["income-noFN"])
        )
        simYear_df["final-demand-noFN"] = (
            simYear_df["demand-pre-debt"] - simYear_df["special-debt-noFN"]
        )
        simYear_df["ckz-preAdj-noFN"] = (
            simYear_df["final-demand-noFN"] - simYear_df["income-noFN"]
        )
        simYear_df.loc[simYear_df["ckz-preAdj-noFN"] < 0, "ckz-preAdj-noFN"] = 0
        return np.abs(
            simYear_df["ckz-preAdj-noFN"].sum()
            - (simYear_df["ckz-preAdj"].sum() - governmentSubsidyTotal)
        )

    res = minimize(ckzPreAdjConstraint, 1)
    extraDebtFactorFound = res.x
    simYear_df["debt-scaling-factor-noFN"] = (
        extraDebtFactorFound * simYear_df["debt-scaling-factor"]
    )
    # simYear_df['special-debt-noFN'] = simYear_df['debt-scaling-factor-noFN'] * \
    #     (simYear_df['demand-pre-debt']-simYear_df['income-noFN']) * \
    #     simYear_df['economic-strength-index-prev3yearavg']
    simYear_df["special-debt-noFN"] = simYear_df["debt-scaling-factor-noFN"] * (
        simYear_df["demand-pre-debt"] - simYear_df["income-noFN"]
    )
    simYear_df["final-demand-noFN"] = (
        simYear_df["demand-pre-debt"] - simYear_df["special-debt-noFN"]
    )
    simYear_df["ckz-preAdj-noFN"] = (
        simYear_df["final-demand-noFN"] - simYear_df["income-noFN"]
    )
    simYear_df.loc[simYear_df["ckz-preAdj-noFN"] < 0, "ckz-preAdj-noFN"] = 0
    assert (
        np.abs(
            1
            - (simYear_df["ckz-preAdj-noFN"].sum() + governmentSubsidyTotal)
            / simYear_df["ckz-preAdj"].sum()
        )
        < 0.001
    )

    # Find a uniform scaling of the adjustment-factors such that with the nouzei undone we still get the right total ckz
    def ckzConstraint(extraFactor):
        simYear_df["ckz-noFN"] = (
            simYear_df["final-demand-noFN"]
            * (1 - extraFactor * simYear_df["adjustment-factor"])
            - simYear_df["income-noFN"]
        )
        simYear_df.loc[simYear_df["ckz-noFN"] < 0, "ckz-noFN"] = 0
        return np.abs(
            simYear_df["ckz-noFN"].sum()
            - (simYear_df["ckz"].sum() - governmentSubsidyTotal)
        )

    res = minimize(ckzConstraint, 1)
    extraAdjFactorFound = res.x
    simYear_df["adjustment-factor-noFN"] = (
        extraAdjFactorFound * simYear_df["adjustment-factor"]
    )
    simYear_df["ckz-noFN"] = (
        simYear_df["final-demand-noFN"] * (1 - simYear_df["adjustment-factor-noFN"])
        - simYear_df["income-noFN"]
    )
    simYear_df.loc[simYear_df["ckz-noFN"] < 0, "ckz-noFN"] = 0
    assert (
        np.abs(
            1
            - (simYear_df["ckz-noFN"].sum() + governmentSubsidyTotal)
            / simYear_df["ckz"].sum()
        )
        < 0.001
    )

    return simYear_df


def do_fn_sim(year_df):
    """Take all the data which includes -noFN cols adds cols which include fn (no suffix)"""

    estimateYear_df = year_df.copy()

    estimateYear_df["income"] = (
        estimateYear_df["income-noFN"] - 0.75 * estimateYear_df["deductions"]
    )

    if shortfall[simReferenceYear + 2] > 0:
        subsidyFraction = 0.5
    else:
        subsidyFraction = 0
    print("Using subsidy fraction", subsidyFraction)
    governmentSubsidyTotal = (
        subsidyFraction * 0.75 * estimateYear_df["deductions"].sum()
    )  # For testing purposes

    # Find a uniform scaling of the debt-scaling-factors without nozei such that with the nouzei we still get the right total ckz-preAdj
    from scipy.optimize import minimize

    def ckzPreAdjConstraint(extraFactor):
        # estimateYear_df['special-debt'] = (extraFactor*estimateYear_df['debt-scaling-factor-noFN']*(
        #     estimateYear_df['demand-pre-debt']-estimateYear_df['income'])*estimateYear_df['economic-strength-index-prev3yearavg'])
        estimateYear_df["special-debt"] = (
            extraFactor
            * estimateYear_df["debt-scaling-factor-noFN"]
            * (estimateYear_df["demand-pre-debt"] - estimateYear_df["income"])
        )

        estimateYear_df["final-demand"] = (
            estimateYear_df["demand-pre-debt"] - estimateYear_df["special-debt"]
        )
        estimateYear_df["ckz-preAdj"] = (
            estimateYear_df["final-demand"] - estimateYear_df["income"]
        )
        estimateYear_df.loc[estimateYear_df["ckz-preAdj"] < 0, "ckz-preAdj"] = 0
        return np.abs(
            estimateYear_df["ckz-preAdj-noFN"].sum()
            - (estimateYear_df["ckz-preAdj"].sum() - governmentSubsidyTotal)
        )

    res = minimize(ckzPreAdjConstraint, 1)
    extraDebtFactorFound = res.x
    estimateYear_df["debt-scaling-factor"] = (
        extraDebtFactorFound * estimateYear_df["debt-scaling-factor-noFN"]
    )

    # estimateYear_df['special-debt'] = estimateYear_df['debt-scaling-factor'] * \
    #     (estimateYear_df['demand-pre-debt']-estimateYear_df['income']
    #      )*estimateYear_df['economic-strength-index-prev3yearavg']
    estimateYear_df["special-debt"] = estimateYear_df["debt-scaling-factor"] * (
        estimateYear_df["demand-pre-debt"] - estimateYear_df["income"]
    )
    estimateYear_df["final-demand"] = (
        estimateYear_df["demand-pre-debt"] - estimateYear_df["special-debt"]
    )
    estimateYear_df["ckz-preAdj"] = (
        estimateYear_df["final-demand"] - estimateYear_df["income"]
    )
    estimateYear_df.loc[estimateYear_df["ckz-preAdj"] < 0, "ckz-preAdj"] = 0
    assert (
        np.abs(
            1
            - (estimateYear_df["ckz-preAdj-noFN"].sum() + governmentSubsidyTotal)
            / estimateYear_df["ckz-preAdj"].sum()
        )
        < 0.001
    )

    # Now do the same for the adjustment factor.
    # Find a uniform scaling of the adjustment-factors such that with the nouzei undone we still get the right total ckz
    def ckzConstraint(extraFactor):
        estimateYear_df["ckz"] = (
            estimateYear_df["final-demand"]
            * (1 - extraFactor * estimateYear_df["adjustment-factor-noFN"])
            - estimateYear_df["income"]
        )
        estimateYear_df.loc[estimateYear_df["ckz"] < 0, "ckz"] = 0
        return np.abs(
            estimateYear_df["ckz-noFN"].sum()
            - (estimateYear_df["ckz"].sum() - governmentSubsidyTotal)
        )

    res = minimize(ckzConstraint, 1)
    extraAdjFactorFound = res.x
    estimateYear_df["adjustment-factor"] = (
        extraAdjFactorFound * estimateYear_df["adjustment-factor-noFN"]
    )
    estimateYear_df["ckz"] = (
        estimateYear_df["final-demand"] * (1 - estimateYear_df["adjustment-factor"])
        - estimateYear_df["income"]
    )
    estimateYear_df.loc[estimateYear_df["ckz"] < 0, "ckz"] = 0
    assert (
        np.abs(
            1
            - (estimateYear_df["ckz-noFN"].sum() + governmentSubsidyTotal)
            / estimateYear_df["ckz"].sum()
        )
        < 0.001
    )

    return estimateYear_df


############### ###
# MUNICIPAL SIM ###
############### ###
sim_df = pd.DataFrame()
simYears = list(range(np.min(fdf.year), np.max(cdf.year) - 2 + 1))
for year in simYears:
    # This is +2 because donations are made in year X, decline in income is reported in year X+1, and then compensation occurs in X+2年度. Implement an estimation of the effect by using the last year's average or something...
    chihoukoufuzeiyear = year + 2
    print(year)
    fdfyear = fdf.loc[fdf.year == year].drop("year", axis=1)
    cdfyear = cdf.loc[cdf.year == chihoukoufuzeiyear].drop("year", axis=1)
    # get rid of the prefectures and tokyo23
    fdfyear = fdfyear.drop(fdfyear.loc[fdfyear["city"] == "prefecture"].index).drop(
        fdfyear.loc[
            (fdfyear["prefecture"] == "東京都") & (fdfyear["city"]).str.contains("区")
        ].index
    )
    cdfyear = cdfyear.drop(
        cdfyear.loc[
            (cdfyear["prefecture"] == "東京都") & (cdfyear["city"]).str.contains("区")
        ].index
    )
    assert len(cdfyear) == len(fdfyear)
    if year in [2017]:
        cdfyear.loc[cdfyear["code"] == "40231", "code"] = "40305"
    year_df = cdfyear.merge(
        fdfyear, on=["prefecture", "code"], validate="one_to_one", suffixes=["", "_fdf"]
    )
    assert len(year_df) == len(fdfyear)
    fdfyear.loc[~fdfyear["code"].isin(year_df["code"]), "code"]
    # fdfyear.loc[fdfyear['city']=='那珂川市','code']
    # cdfyear.loc[cdfyear['city']=='那珂川市','code']
    # fdf.loc[fdf['city']=='那珂川市','code']
    # cdf.loc[cdf['city']=='那珂川市','code']
    # cdf.loc[cdf['code'] == '40305']
    simYear_df = undo_fn_sim(year_df)

    tokyo23codes = [str(13101 + i) for i in range(23)]
    tokyo23df = pd.DataFrame(tokyo23codes, columns=["code"])
    tokyo23df["ckz"] = 0
    tokyo23df["ckz-noFN"] = 0
    tokyo23df["special-debt"] = 0
    tokyo23df["special-debt-noFN"] = 0
    tokyo23df["prefecture"] = "東京都"
    tokyo23df["economic-strength-index-prev3yearavg"] = np.nan
    simYear_df = pd.concat([simYear_df, tokyo23df])
    simYear_df["year"] = int(year)

    sim_df = pd.concat([sim_df, simYear_df])

estimate_df = pd.DataFrame()
estimateYears = list(range(np.max(cdf.year) - 1, np.max(fdf.year) + 1))
simReferenceYear = np.max(sim_df["year"])

# 2016,2017,
# for simReferenceYear in [2018,2019,2020]:
#     print(simReferenceYear)
for year in estimateYears:
    referenceSim = (
        sim_df.loc[
            (sim_df["year"] == simReferenceYear) & ~sim_df["code"].isin(tokyo23codes)
        ]
        .reset_index(drop=True)
        .drop("year", axis=1)[
            [
                "prefecture",
                "code",
                "ckz-noFN",
                "ckz-preAdj-noFN",
                "special-debt-noFN",
                "income-noFN",
                "demand-pre-debt",
                "final-demand-noFN",
                "debt-scaling-factor-noFN",
                "adjustment-factor-noFN",
                "economic-strength-index-prev3yearavg",
            ]
        ]
    )
    fdfyear = fdf.loc[fdf.year == year].drop("year", axis=1)
    # get rid of the prefectures and tokyo23
    fdfyear = (
        fdfyear.drop(fdfyear.loc[fdfyear["city"] == "prefecture"].index)
        .drop(
            fdfyear.loc[
                (fdfyear["prefecture"] == "東京都") & (fdfyear["city"]).str.contains("区")
            ].index
        )
        .reset_index(drop=True)
    )
    year_df = referenceSim.merge(
        fdfyear, on=["prefecture", "code"], validate="one_to_one", suffixes=["", "_fdf"]
    )
    assert len(fdfyear) == len(year_df)

    estimateYear_df = do_fn_sim(year_df)

    tokyo23codes = [str(13101 + i) for i in range(23)]
    tokyo23df = pd.DataFrame(tokyo23codes, columns=["code"])
    tokyo23df["ckz"] = 0
    tokyo23df["ckz-noFN"] = 0
    tokyo23df["special-debt"] = 0
    tokyo23df["special-debt-noFN"] = 0
    tokyo23df["prefecture"] = "東京都"
    tokyo23df["economic-strength-index-prev3yearavg"] = np.nan
    estimateYear_df = pd.concat([estimateYear_df, tokyo23df])
    estimateYear_df["year"] = int(year)

    estimate_df = pd.concat([estimate_df, estimateYear_df])

    # print('typical effect', np.median(np.abs(estimateYear_df['ckz-noFN']-estimateYear_df['ckz']))/10**8)

sim_df = sim_df.drop(
    [
        "deficit-or-surplus",
        "total-debt-payback",
        "ckz-pre-rev",
        "economic-strength-index",
        "city_fdf",
    ],
    axis=1,
)

ckz_muni_sim_df = pd.concat([sim_df, estimate_df])

########################
###### PREF SIM ########
########################

sim_df = pd.DataFrame()
simYears = list(range(np.min(fdf.year), np.max(cdf.year) - 2 + 1))
for year in simYears:
    # This is +2 because donations are made in year X, decline in income is reported in year X+1, and then compensation occurs in X+2年度. Implement an estimation of the effect by using the last year's average or something...
    chihoukoufuzeiyear = year + 2
    print(year)
    fdfyear = fdf.loc[fdf.year == year].drop("year", axis=1)
    cdfyear = cdf_pref.loc[cdf_pref.year == chihoukoufuzeiyear].drop("year", axis=1)
    # keep only the prefectures, get rid of tokyo
    fdfyear = fdfyear.drop(
        fdfyear.loc[
            (fdfyear["city"] != "prefecture") | (fdfyear["prefecture"] == "東京都")
        ].index
    )
    cdfyear = cdfyear.drop(cdfyear.loc[(cdfyear["prefecture"] == "東京都")].index)

    year_df = cdfyear.merge(
        fdfyear, on=["prefecture"], validate="one_to_one", suffixes=["", "_fdf"]
    )
    assert len(cdfyear) == len(fdfyear)
    assert len(year_df) == len(fdfyear)

    simYear_df = undo_fn_sim(year_df)

    tokyocode = ["13000"]
    tokyodf = pd.DataFrame(tokyocode, columns=["code_fdf"])
    tokyodf["ckz"] = 0
    tokyodf["ckz-noFN"] = 0
    tokyodf["special-debt"] = 0
    tokyodf["special-debt-noFN"] = 0
    tokyodf["prefecture"] = "東京都"
    tokyodf["economic-strength-index-prev3yearavg"] = np.nan
    simYear_df = pd.concat([simYear_df, tokyodf])

    simYear_df["year"] = int(year)

    print(
        "typical effect",
        np.median(np.abs(simYear_df["ckz-noFN"] - simYear_df["ckz"])) / 10**8,
    )

    sim_df = pd.concat([sim_df, simYear_df])

estimate_df = pd.DataFrame()
estimateYears = list(range(np.max(cdf.year) - 1, np.max(fdf.year) + 1))
simReferenceYear = np.max(sim_df["year"])

# 2016,2017,
# for simReferenceYear in [2018,2019,2020]:
#     print(simReferenceYear)
for year in estimateYears:
    referenceSim = (
        sim_df.loc[(sim_df["year"] == simReferenceYear)]
        .reset_index(drop=True)
        .drop("year", axis=1)[
            [
                "prefecture",
                "code_fdf",
                "ckz-noFN",
                "ckz-preAdj-noFN",
                "special-debt-noFN",
                "income-noFN",
                "demand-pre-debt",
                "final-demand-noFN",
                "debt-scaling-factor-noFN",
                "adjustment-factor-noFN",
                "economic-strength-index-prev3yearavg",
            ]
        ]
    )
    fdfyear = fdf.loc[fdf.year == year].drop("year", axis=1)
    # keep only the prefectures, get rid of tokyo
    fdfyear = fdfyear.drop(
        fdfyear.loc[
            (fdfyear["city"] != "prefecture") | (fdfyear["prefecture"] == "東京都")
        ].index
    )
    cdfyear = cdfyear.drop(cdfyear.loc[(cdfyear["prefecture"] == "東京都")].index)
    year_df = referenceSim.merge(
        fdfyear, on=["prefecture"], validate="one_to_one", suffixes=["", "_fdf"]
    )
    assert len(fdfyear) == len(cdfyear)
    assert len(year_df) == len(fdfyear)

    estimateYear_df = do_fn_sim(year_df)

    tokyocode = ["13000"]
    tokyodf = pd.DataFrame(tokyocode, columns=["code_fdf"])
    tokyodf["ckz"] = 0
    tokyodf["ckz-noFN"] = 0
    tokyodf["special-debt"] = 0
    tokyodf["special-debt-noFN"] = 0
    tokyodf["prefecture"] = "東京都"
    tokyodf["economic-strength-index-prev3yearavg"] = np.nan
    estimateYear_df = pd.concat([estimateYear_df, tokyodf])

    estimateYear_df["year"] = int(year)

    estimate_df = pd.concat([estimate_df, estimateYear_df])

    print(
        "typical effect",
        np.median(np.abs(estimateYear_df["ckz-noFN"] - estimateYear_df["ckz"]))
        / 10**8,
    )

sim_df = sim_df.drop(
    [
        "total-debt-payback",
        "economic-strength-index",
    ],
    axis=1,
)

ckz_pref_sim_df = pd.concat([sim_df, estimate_df])

if __name__ == "__main__":

    print("municipal stats")
    for year in ckz_muni_sim_df.year.unique():
        print(year)
        df = ckz_muni_sim_df.loc[ckz_muni_sim_df["year"] == year]
        # In principle if the money is subsidized most places should get more ckz with FN than without, since FN decreases their income.
        myFNeffect = df["ckz"] - df["ckz-noFN"]

        print(len(df.loc[myFNeffect > 0]), " places would lose CKZ money if FN ends")
        print(
            "their ESI is ",
            df.loc[myFNeffect > 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[myFNeffect < 0]), " places would gain CKZ money if FN ends")
        print(
            "their ESI is ",
            df.loc[myFNeffect < 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[myFNeffect == 0]), " places + tokyo would see no difference")
        print(
            "their ESI is ",
            df.loc[myFNeffect == 0, "economic-strength-index-prev3yearavg"].mean(),
        )

        totalFNeffect = df["netgainminusdeductions"] + df["ckz"] - df["ckz-noFN"]

        print(
            len(df.loc[totalFNeffect > 0]), " places would lose total money if FN ends"
        )
        print(
            "their ESI is ",
            df.loc[totalFNeffect > 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(
            len(df.loc[totalFNeffect < 0]), " places would gain total money if FN ends"
        )
        print(
            "their ESI is ",
            df.loc[totalFNeffect < 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[totalFNeffect == 0]), " places would see no difference")
        print(
            "their ESI is ",
            df.loc[totalFNeffect == 0, "economic-strength-index-prev3yearavg"].mean(),
        )

        print("typical effect", np.median(np.abs(df["ckz-noFN"] - df["ckz"])) / 10**8)

    print("prefectural stats")
    for year in ckz_pref_sim_df.year.unique():
        print(year)
        df = ckz_pref_sim_df.loc[ckz_pref_sim_df["year"] == year]
        # In principle if the money is subsidized most places should get more ckz with FN than without, since FN decreases their income.
        myFNeffect = df["ckz"] - df["ckz-noFN"]

        print(len(df.loc[myFNeffect > 0]), " places would lose CKZ money if FN ends")
        print(
            "their ESI is ",
            df.loc[myFNeffect > 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[myFNeffect < 0]), " places would gain CKZ money if FN ends")
        print(
            "their ESI is ",
            df.loc[myFNeffect < 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[myFNeffect == 0]), " places would see no difference")
        print(
            "their ESI is ",
            df.loc[myFNeffect == 0, "economic-strength-index-prev3yearavg"].mean(),
        )

        totalFNeffect = df["netgainminusdeductions"] + df["ckz"] - df["ckz-noFN"]

        print(
            len(df.loc[totalFNeffect > 0]), " places would lose total money if FN ends"
        )
        print(
            "their ESI is ",
            df.loc[totalFNeffect > 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(
            len(df.loc[totalFNeffect < 0]), " places would gain total money if FN ends"
        )
        print(
            "their ESI is ",
            df.loc[totalFNeffect < 0, "economic-strength-index-prev3yearavg"].mean(),
        )
        print(len(df.loc[totalFNeffect == 0]), " places would see no difference")
        print(
            "their ESI is ",
            df.loc[totalFNeffect == 0, "economic-strength-index-prev3yearavg"].mean(),
        )

        print("typical effect", np.median(np.abs(df["ckz-noFN"] - df["ckz"])) / 10**8)

    # Here I want to do some checks of ckz time trends to see whether its a good assumption to assume ckz in R5 is like R4,
    # and how minimal can I make the assumptions
    ###
    cb = CircusBoy()

    # First some global checks
    fig, ax = cb.handlers()
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['ckz'],label='ckz')
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['income'],label='income')
    ax.plot(
        ckz_muni_sim_df.year.unique(),
        ckz_muni_sim_df.groupby("year").sum()["income-noFN"],
        label="income no FN",
    )

    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['final-demand'],label='final-demand')
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['demand-pre-debt'],label='demand-pre-debt')
    ax.legend()
    fig.savefig("./simplots/muni-ckz-over-time.pdf")
    plt.close("all")

    # Income focused checks
    fig, ax = cb.handlers()
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['ckz'],label='ckz')
    ax.plot(
        ckz_muni_sim_df.year.unique(),
        ckz_muni_sim_df.groupby("year").sum()["income-noFN"]
        - ckz_muni_sim_df.groupby("year").sum()["income"],
        label="income-noFN-minus-income",
    )
    ax.plot(
        ckz_muni_sim_df.year.unique(),
        0.75 * ckz_muni_sim_df.groupby("year").sum()["deductions"],
        ls="--",
        label="0.75 deductions",
    )
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['final-demand'],label='final-demand')
    # ax.plot(ckz_muni_sim_df.year.unique(),ckz_muni_sim_df.groupby('year').sum()['demand-pre-debt'],label='demand-pre-debt')
    ax.legend()
    fig.savefig("./simplots/muni-income-over-time.pdf")
    plt.close("all")

    # Now pick a random city and do some checks
    code = np.random.choice(ckz_muni_sim_df["code"].unique())
    citydf = ckz_muni_sim_df.loc[ckz_muni_sim_df["code"] == code].sort_values("year")
    print(citydf["city"].values[0], citydf["prefecture"].values[0])
    fig, ax = cb.handlers()
    # ax.plot(citydf.year,citydf['ckz'],label='ckz')
    # ax.plot(citydf.year,citydf['income'],label='income')
    # ax.plot(citydf.year,citydf['final-demand'],label='final-demand')
    # ax.plot(citydf.year,citydf['demand-pre-debt'],label='demand-pre-debt')
    # ax.plot(citydf.year,citydf['adjustment-factor'],label='adjustment-factor')
    ax.plot(citydf.year, citydf["debt-scaling-factor"], label="debt-scaling-factor")
    ax.legend()
    fig.savefig("./simplots/random-city-over-time.pdf")
    plt.close("all")

    # Now pick a few random cities and do some checks
    codes = np.random.choice(ckz_muni_sim_df["code"].unique(), 10, replace=False)
    fig, ax = cb.handlers()
    for code in codes:
        citydf = ckz_muni_sim_df.loc[ckz_muni_sim_df["code"] == code].sort_values(
            "year"
        )
        print(citydf["city"].values[0], citydf["prefecture"].values[0])
        # ax.plot(citydf.year,citydf['ckz']/citydf['ckz'].values[0],label='ckz', alpha=.5)
        # ax.plot(citydf.year,citydf['income']/citydf['income'].values[0],label='income', alpha=.5)
        # ax.plot(citydf.year,citydf['final-demand']/citydf['final-demand'].values[0],label='final-demand', alpha=.5)
        # ax.plot(citydf.year,citydf['demand-pre-debt']/citydf['demand-pre-debt'].values[0],label='demand-pre-debt', alpha=.5)
        # ax.plot(citydf.year,citydf['debt-scaling-factor']/citydf['debt-scaling-factor'].values[0],label='debt-scaling-factor', alpha=.5)
        ax.plot(
            citydf.year,
            citydf["debt-scaling-factor"].values
            / (ckz_muni_sim_df.groupby("year").mean()["debt-scaling-factor"]).values,
            label="debt-scaling-factor",
            alpha=0.5,
        )
        # ax.plot(citydf.year,citydf['adjustment-factor']/citydf['adjustment-factor'].values[0],label='adjustment-factor', alpha=.5)
    # ax.legend()
    fig.savefig("./simplots/random-cities-over-time.pdf")
    plt.close("all")

    # df = ckz_pref_sim_df.loc[(ckz_pref_sim_df['city']=='prefecture') &(ckz_pref_sim_df['prefecture']!='東京都')]
    # df['dummy'] = (df['profit-from-ckz']-df['deductions'])/df['economic-strength-index']
    # #### First show what happened in 2020, which is not a seppan year
    # year_df = df.loc[df['year']==2020]

    # ## Trying to understand what determines which places again and lose money from the ckz effect

    # fig, ax = cb.handlers()
    # ax.scatter(year_df['deductions'], year_df['profit-from-ckz'])
    # fig.savefig(PLOT_FOLDER+'ckz-profit-vs-deductions')
    # plt.close('all')

    # import plotly.express as px
    # import plotly.graph_objects as go
    # year_df = df.loc[df['year'].isin([2018,2020,2021])]
    # # fig = px.line(year_df, x='deductions',y='profit-from-ckz', color='prefecture',hover_data=['prefecture', 'year','economic-strength-index'], template='simple_white',markers=True)
    # fig = px.line(year_df, x='economic-strength-index',y='dummy', color='prefecture',hover_data=['prefecture', 'year','economic-strength-index'], template='simple_white',markers=True)

    # fig.write_html(PLOT_FOLDER+'ckz-profit-vs-deductions.html')
    # plt.close('all')

else:
    ckz_muni_sim_df = ckz_muni_sim_df[
        [
            "prefecture",
            "code",
            "year",
            "ckz",
            "ckz-noFN",
            "demand-pre-debt",
            "income",
            "income-noFN",
            "economic-strength-index-prev3yearavg",
            "debt-scaling-factor",
            "special-debt",
            "special-debt-noFN",
            "special-debt-payback",
        ]
    ]
    ckz_pref_sim_df = ckz_pref_sim_df[
        [
            "prefecture",
            "code_fdf",
            "year",
            "ckz",
            "ckz-noFN",
            "demand-pre-debt",
            "income",
            "income-noFN",
            "economic-strength-index-prev3yearavg",
            "debt-scaling-factor",
            "special-debt",
            "special-debt-noFN",
            "special-debt-payback",
        ]
    ].rename({"code_fdf": "code"}, axis=1)
    ckz_sim_df = pd.concat([ckz_muni_sim_df, ckz_pref_sim_df])
