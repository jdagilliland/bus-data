library(data.table)
library(ggplot2)
library(lubridate)

bus.data <- fread("~/tmp/bus-data.csv", colClasses = c("character","character","character"))
setnames(bus.data, c("V1", "V2", "V3"), c("date", "departure", "arrival"))
bus.data[, departure:=paste(substr(departure, 1, 2), substr(departure,3,4), sep=":")]
bus.data[, arrival:=paste(substr(arrival, 1, 2), substr(arrival,3,4), sep=":")]
bus.data[, datetime.departure:=ymd_hm(paste(date, departure, sep="_"))]
bus.data[, datetime.arrival:=ymd_hm(paste(date, arrival, sep="_"))]
bus.data[, duration:=datetime.arrival - datetime.departure]

dropDate <- function(x) {
    3600 * hour(x) + 60 * minute(x) + second(x)
}
dropDate2 <- function(x) {
  as.numeric(x - as.Date(x))
}

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

ggsave("plot-durations.png", plot=plot.durations, units="in", height=8.5, width=4)