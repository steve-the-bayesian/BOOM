import unittest
import pandas as pd

import pdb

from BayesBoom.bsts import (
    # Bsts,
    # RegressionHolidayStateModel,
    FixedDateHoliday,
    LastWeekdayInMonthHoliday,
    # NthWeekdayInMonthHoliday,
    EasterSunday,
    USDaylightSavingsTimeBegins,
    USDaylightSavingsTimeEnds,
)


class TestHolidays(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
        # delete_if_present("bsts_llt.pkl")

    def test_boom_holidays(self):
        easter = EasterSunday()
        self.assertAlmostEqual(
            easter.date(2021),
            pd.Timestamp(year=2021, month=4, day=4))
        self.assertAlmostEqual(
            easter.date(2020),
            pd.Timestamp(year=2020, month=4, day=12))

        dst_begin = USDaylightSavingsTimeBegins()
        self.assertAlmostEqual(
            dst_begin.date(2021),
            pd.Timestamp(year=2021, month=3, day=14))

        dst_end = USDaylightSavingsTimeEnds()
        self.assertAlmostEqual(
            dst_end.date(2020),
            pd.Timestamp(year=2020, month=11, day=1))

        july4 = FixedDateHoliday(month=7, day=4)
        self.assertAlmostEqual(
            july4.date(2020),
            pd.Timestamp(year=2020, month=7, day=4))

        memorial_day = LastWeekdayInMonthHoliday(1, 5)
        self.assertAlmostEqual(
            memorial_day.date(2020),
            pd.Timestamp(year=2020, month=5, day=25))


class TestRegressionHolidayStateModel(unittest.TestCase):
    pass


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestHolidays()
    # rig = TestGaussianTimeSeries()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_boom_holidays()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
