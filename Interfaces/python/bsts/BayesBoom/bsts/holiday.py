import BayesBoom.boom as boom
import BayesBoom.R as R

from abc import ABC, abstractmethod

HOLIDAY_REGISTRY = {}

# TODO update holiday registry by country.

days_in_month = (-1,
                 31,  # January
                 28,  # February
                 31,  # March
                 30,  # April
                 31,  # May
                 30,  # June
                 31,  # July
                 31,  # August
                 30,  # September
                 31,  # October
                 30,  # November
                 31,  # December
                 )


def register_holiday(name, holiday_object):
    global HOLIDAY_REGISTRY
    HOLIDAY_REGISTRY[name] = holiday_object


class Holiday(ABC):
    """
    """
    @abstractmethod
    def boom(self):
        """
        Return a boom holiday corresponding to the information in this holiday
        object.
        """

    @property
    def maximum_window_width(self):
        """
        Holidays exert influence before or after the day itself.  The
        maximum_window_width is the number of days of infludence a holiday has
        each time it occurs.
        """
        return self._boom_holiday.maximum_window_width

    def active(self, arbitrary_date):
        """
        Returns True iff 'arbitrary_date' falls inisde the influence window for
        the holiday.
        """
        return self._boom_holiday.active(R.to_boom_date(arbitrary_date))

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_boom_holiday"]

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._boom_holiday = self.boom()


class SingleDayHoliday(Holiday):
    """
    A SingleDayHoliday is a holiday associated with a specific date.  Its
    influence can extend beyond that date, but (e.g.) February 14 is
    Valentine's day.  Most Holidays are SingleDayHolidays, some religious
    holidays (e.g. Passover) and some sporting events (e.g. the Olympics or the
    World Cup) are not.
    """

    def date_on_or_after(self, date):
        """
        Either 'date' or the first instance of the holiday after 'date'.
        """
        ans = self._boom_holiday.date_on_or_after(R.to_boom_date(date))
        return R.to_pd_timestamp(ans)

    def date_on_or_before(self, date):
        """
        Either 'date' or the first instance of the holiday before 'date'.
        """
        return R.to_pd_timestamp(self._boom_holiday.date_on_or_before(
            R.to_boom_date(date)))

    def nearest(self, date):
        """
        The date of the closest instance of this holiday to 'date'.
        """
        return R.to_pd_timestamp(self._boom_holiday.nearest(
            R.to_boom_date(date)))


class OrdinaryAnnualHoliday(SingleDayHoliday):
    """
    An OrdinaryAnnualHoliday is a Holiday that occurs once per year, with a
    fixed-sized window of influence.  An OrdinaryAnnualHoliday keeps track of
    two integers: days_before and days_after, that define its influence window.
    """
    def __init__(self, days_before: int = 1, days_after: int = 1):
        if not days_before >= 0:
            raise Exception("'days_before' must be non-negative.")
        if not days_after >= 0:
            raise Exception("'days_after' must be non-negative.")
        self._days_before = int(days_before)
        self._days_after = int(days_after)

    def date(self, year):
        return R.to_pd_timestamp(self._boom_holiday.date(int(year)))


class FixedDateHoliday(OrdinaryAnnualHoliday):
    """
    A holiday that occurs on the same calendar date each year.
    """
    def __init__(self,
                 month: int,
                 day: int,
                 days_before: int = 1,
                 days_after: int = 1):
        super().__init__(days_before, days_after)
        if month < 1 or month > 12:
            raise Exception("'month' must be between 1 and 12.")
        if day < 1 or day > days_in_month[month]:
            raise Exception(
                f"For month {month}, the day argument must be between 1 "
                f"and {days_in_month[month]}."
            )
        self._month = int(month)
        self._day = int(day)
        self._boom_holiday = self.boom()

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.FixedDateHoliday(
            self._month,
            self._day,
            self._days_before,
            self._days_after)


class NthWeekdayInMonthHoliday(OrdinaryAnnualHoliday):
    """
    A holidy that occurs each year on the same day of the week, defined as the
    Nth such dat in a month.  For example, US Labor Day is the first Monday in
    September.
    """
    def __init__(self,
                 which_week: int,
                 day_of_week: int,
                 month: int,
                 days_before: int = 1,
                 days_after: int = 1):
        super().__init__(days_before, days_after)
        if which_week < 1 or which_week > 5:
            raise Exception("which_week must be between 1 and 5.")
        if month < 1 or month > 12:
            raise Exception("month must be between 1 and 12.")
        if day_of_week < 1 or day_of_week > 7:
            raise Exception("day_of_week must be between 1 and 7.")
        self._which_week = which_week
        self._day_of_week = day_of_week
        self._month = month
        self._boom_holiday = self.boom()

    def boom(self):
        # TODO(steve): Document the mapping between int's and days.
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.NthWeekdayInMonthHoliday(
            self._which_week,
            self._day_of_week,
            self._month,
            self._days_before,
            self._days_after)


class LastWeekdayInMonthHoliday(OrdinaryAnnualHoliday):
    """
    A holiday that occurs each year on the final weekday in a given month.  For
    example, US Memorial Day is the last Monday in May.
    """
    def __init__(self,
                 day_of_week: int,
                 month: int,
                 days_before: int = 1,
                 days_after: int = 1):
        super().__init__(days_before, days_after)
        if month < 1 or month > 12:
            raise Exception("month must be between 1 and 12")
        if day_of_week < 1 or day_of_week > 7:
            raise Exception("day_of_week must be between 1 and 7")
        self._month = int(month)
        self._day_of_week = int(day_of_week)
        self._boom_holiday = self.boom()

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.LastWeekdayInMonthHoliday(
            self._day_of_week,
            self._month,
            self._days_before,
            self._days_after)


class DateRangeHoliday(Holiday):
    """
    An irregular holiday, for which the influence window must be handled
    manually.
    """
    def __init__(self, start, end):
        self._start = start
        self._end = end
        self._boom_holiday = self.boom()

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        start_days = [R.to_boom_date(x) for x in self._start]
        end_days = [R.to_boom_date(x) for x in self._end]

        return boom.DateRangeHoliday(start_days, end_days)


# ======================================================================
# Some holidays that require smarts from BOOM.
# ======================================================================

class EasterSunday(OrdinaryAnnualHoliday):
    def __init__(self, days_before: int = 1, days_after: int = 1):
        self._days_before = int(days_before)
        self._days_after = int(days_after)
        self._boom_holiday = self.boom()

    def date(self, year):
        return R.to_pd_timestamp(self._boom_holiday.date(int(year)))

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.EasterSunday(
            self._days_before, self._days_after)


class USDaylightSavingsTimeBegins(OrdinaryAnnualHoliday):
    def __init__(self, days_before: int = 1, days_after: int = 1):
        self._days_before = int(days_before)
        self._days_after = int(days_after)
        self._boom_holiday = self.boom()

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.USDaylightSavingsTimeBegins(
            self._days_before, self._days_after)


class USDaylightSavingsTimeEnds(OrdinaryAnnualHoliday):
    def __init__(self, days_before: int = 1, days_after: int = 1):
        self._days_before = int(days_before)
        self._days_after = int(days_after)
        self._boom_holiday = self.boom()

    def boom(self):
        if hasattr(self, "_boom_holiday"):
            return self._boom_holiday
        return boom.USDaylightSavingsTimeEnds(
            self._days_before, self._days_after)


class HolidayFactory:

    def create_boom_holiday(self,
                            holiday,
                            days_before: int = 1,
                            days_after: int = 1):
        if isinstance(holiday, Holiday):
            return holiday.boom()
        elif isinstance(holiday, str) and holiday in HOLIDAY_REGISTRY.keys():
            if holiday in us_named_holidays:
                return boom.create_named_holiday(
                    holiday, int(days_before), int(days_after))
        else:
            raise Exception(f"Unrecognized holiday: {holiday}.")


us_named_holidays = [
    "NewYearsDay",
    "MartinLutherKingDay",
    "SuperBowlSunday",
    "PresidentsDay",
    "ValentinesDay",
    "SaintPatricksDay",
    "USDaylightSavingsTimeBegins",
    "USDaylightSavingsTimeEnds",
    "EasterSunday",
    "USMothersDay",
    "IndependenceDay",
    "LaborDay",
    "ColumbusDay",
    "Halloween",
    "Thanksgiving",
    "MemorialDay",
    "VeteransDay",
    "Christmas",
]
