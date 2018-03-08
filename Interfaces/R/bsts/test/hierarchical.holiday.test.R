
turkish <- scan("turkey_elec.csv")
dates <- seq(as.Date("2000-01-01"), len = length(turkish), by = "day")
turkish <- zoo(turkish, dates)

ss <- AddSemilocalLinearTrend(list(), turkish)
ss <- AddSeasonal(ss, turkish, nseasons = 52, season.duration = 7)
ss <- AddSeasonal(ss, turkish, nseasons = 7)
base.model <- bsts(turkish, ss, niter = 1000)

EidAlFitr <- DateRangeHoliday("EidAlFit"
                              start = as.Date(c("2000-12-27",
                                                "2001-12-16",
                                                "2002-12-05",
                                                "2003-11-25",
                                                "2004-11-13",
                                                "2005-11-02",
                                                "2006-10-23",
                                                "2007-10-12",
                                                "2008-10-01",
                                                "2009-09-20",
                                                "2010-09-10")),
                              end = as.Date(c("2000-12-30",
                                              "2001-12-19",
                                              "2002-12-08",
                                              "2003-11-28",
                                              "2004-11-16",
                                              "2005-11-05",
                                              "2006-10-26",
                                              "2007-10-15",
                                              "2008-10-04",
                                              "2009-09-23",
                                              "2010-09-13")))

EidAlAdha <- DateRangeHoliday("EidAlAdha",
                              start = as.Date(c("2000-03-15",
                                                "2001-03-05",
                                                "2002-02-22",
                                                "2003-02-11",
                                                "2004-02-01",
                                                "2005-01-20",
                                                "2006-12-30",
                                                "2007-12-19",
                                                "2008-12-08",
                                                "2009-11-27",
                                                "2010-11-16")),
                              end = as.Date(c("2000-03-19",
                                              "2001-03-09",
                                              "2002-02-26",
                                              "2003-02-15",
                                              "2004-02-05",
                                              "2005-01-24",
                                              "2007-01-03",
                                              "2007-12-23",
                                              "2008-12-12",
                                              "2009-12-01",
                                              "2010-11-20")))
