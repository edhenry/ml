# This is a script written to convert .binetflow files to csv files
# Run this scripting using Rscript at the command line and pass in
# the name of the .binetflow file that you'd like to convert as an
# argument.
#
# ex: Rstudio nf_cleanup foobar.binetflow

# Load default R methods
library(methods)

# Save command line arguments as variables
args <- commandArgs(trailingOnly = TRUE)

setClass("myPosixCt")

# Set StartTime vector as POSIX compliant
setAs("character", "myPosixCt", function(from) as.POSIXct(from, format = "%Y/%m/%d %H:%M:%OS"))

# Set environment options for POSIX time conversion to trailng millionths (6 spaces) of a second
options(set.seconds="6")

# Read in the .binetflow file and convert corresponding vectors to listed data types
flowdata = read.csv(args[1], colClasses = c("myPosixCt", "numeric", "factor", "factor","factor","factor","factor","factor","factor","factor","factor","numeric", "numeric", "numeric", "factor"), strip.white = TRUE, sep = ',')

# Var containing vectors that we would like dropped from the data
drops <- c("X", "StartTime", "Dur", "Dir", "State", "sTos", "dTos", "Label")

# Drop the corresponding vectors from the data frame
flowdata <- flowdata[,!(names(flowdata) %in% drops)]

#Write out the resulting data frame to a CSV file
write.csv(flowdata, file="flowdata.csv")

# Stop script
stop(message("File converted!"))


