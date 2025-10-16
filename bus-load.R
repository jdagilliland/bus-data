library(data.table)
library(ggplot2)
library(lubridate)
library(scales)
library(purrr)

bus.data.files <- c("data/nashua.csv", "data/north-londonderry.csv")
bus.data.tables <- map(bus.data.files,
                       ~ fread(.x,
                               colClasses = c("character","character","character"),
                               col.names = c("date", "departure", "arrival")))
bus.data.nashua <- bus.data.tables[[1]][, station := "nashua"]
bus.data.nlondonderry <- bus.data.tables[[2]][, station := "north-londonderry"]
bus.data <- rbind(bus.data.nashua, bus.data.nlondonderry)

bus.data[, departure:=paste(substr(departure, 1, 2), substr(departure,3,4), sep=":")]
bus.data[, arrival:=paste(substr(arrival, 1, 2), substr(arrival,3,4), sep=":")]
bus.data[, datetime.departure:=ymd_hm(paste(date, departure, sep="_"))]
# bus.data[, time.departure:=hm(departure)]
# bus.data[, time.arrival:=hm(arrival)]
bus.data[, datetime.arrival:=ymd_hm(paste(date, arrival, sep="_"))]
bus.data[, duration:=datetime.arrival - datetime.departure]

dropDate <- function(x) {
    3600 * hour(x) + 60 * minute(x) + second(x)
}
dropDate2 <- function(x) {
  as.numeric(x - as.Date(x))
}

# From here on only plots.

plot.durations <- ggplot(bus.data) + aes(x=departure, y=duration) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)
plot.arrivals.1 <- ggplot(bus.data) +
    aes(x=departure, y=as.Date(hm(arrival), origin=lubridate::origin)) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)
plot.arrivals.2 <- ggplot(bus.data) +
    aes(x=departure, y=dropDate2(datetime.arrival)) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3) +
    scale_y_datetime(labels=date_format("%H:%M"))
plot.arrivals.3 <- ggplot(bus.data) +
  aes(x=departure, y=dropDate2(datetime.arrival)) +
  geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)

plot.arrivals.1.inbound <- ggplot(bus.data[departure %in%
                                  c("05:40", "06:20", "06:40", "07:20",
                                    "08:00", "09:00", "10:30")]) +
    aes(x=departure, y=as.Date(hm(arrival), origin=lubridate::origin)) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)

plot.arrivals.2.inbound <- ggplot(bus.data[hm(departure) < hm("12:00")]) +
    aes(x=departure, y=as.Date(hm(arrival), origin=lubridate::origin), color=station) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)

plot.arrivals.1.outbound <- ggplot(bus.data[hm(departure) > hm("12:00")]) +
    aes(x=departure, y=as.Date(hm(arrival), origin=lubridate::origin), color=station) +
    geom_boxplot(outlier.color = NA, fill=NA) + geom_jitter(width=0.3)

ggsave("plot-durations.png", plot=plot.durations, units="in", height=8.5, width=6)
