setwd("C:/Users/escob/Desktop/Code/M2 - MBDS/R/Projet-Visu")
songs = read.csv("Spotify Most Streamed Songs.csv")
View(songs)
songs = read.csv("Spotify Most Streamed Songs.csv",header=T,stringsAsFactors = T)
View(songs)
View(songs)
View(songs)
colnames(songs)
print(colnames(songs))
colnames(songs)
songs$Streams = as.numeric(songs$Streams)
colnames(songs)
gc()
View(songs)
songs[1,]
songs <- subset(songs, select = -cover_url)
artists <- read.csv("artists.csv",header=T,stringsAsFactors = T)
View(artists)
View(artists)
artists <- subset(artists, select = -c(ambiguous_artist,scrobbles_lastfm,listeners_lastfm))
artists <- subset(artists, select = -c(mbids,ambiguous_artist,scrobbles_lastfm,listeners_lastfm))
artists <- subset(artists, select = -c(mbid,ambiguous_artist,scrobbles_lastfm,listeners_lastfm))
artists <- subset(artists, select = -c(mbid))
write.csv(artists, "artists_cleaned.csv", row.names = FALSE)
songs <- read.csv("Spotify Most Streamed Songs.csv",header=T,stringsAsFactors = T)
songs <- subset(songs, select = -cover_url)
artists <- read.csv("artists_cleaned.csv",header=T,stringsAsFactors = T)
View(artists)
artists[1,]
artists[1,]
View(artists)
#remove all artists that are not in the songs dataset
artists <- artists[artists$artist %in% songs$artist,]
artists <- read.csv("artists_cleaned.csv",header=T,stringsAsFactors = T)
is_ascii <- function(x) {
# Tente de convertir en ASCII en supprimant les caractères non convertibles
x_ascii <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = "")
# Vérifie si la longueur après conversion est la même
nchar(x_ascii) == nchar(x) & !is.na(x_ascii)
}
is_ascii <- function(x) {
grepl("^[\x00-\x7F]+$", x)
is_ascii <- function(x) {
grepl("^[\x00-\x7F]+$", x)
is_ascii <- function(x) {
grepl("^[\x00-\x7F]+$", x)
is_ascii <- function(x) {
# Tente de convertir en ASCII en supprimant les caractères non convertibles
x_ascii <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = "")
# Vérifie si la longueur après conversion est la même
nchar(x_ascii) == nchar(x) & !is.na(x_ascii)
}
ascii_indices <- is_ascii(artists$artist_mb)
ascii_indices <- is_ascii(artists$artist_mb)
class(artists$artist_mb)
artists$artist_mb <- as.character(artists$artist_mb)
ascii_indices <- is_ascii(artists$artist_mb)
artists_ascii <- artists[ascii_indices, ]
n_total <- nrow(artists)
n_ascii <- nrow(artists_ascii)
artists_ascii$artist_mb <- trimws(artists_ascii$artist_mb)
songs$artist.s._name <- trimws(songs$artist.s._name)
artists_ascii$artist_mb <- tolower(artists_ascii$artist_mb)
songs$artist.s._name <- tolower(songs$artist.s._name)
ascii_artist_names <- unique(artists_ascii$artist_mb)
songs_ascii <- songs[tolower(songs$artist.s._name) %in% ascii_artist_names, ]
head(songs_ascii)
View(songs_ascii)
View(songs_ascii)
View(artists_ascii)
songs <- read.csv("Spotify Most Streamed Songs.csv",header=T,stringsAsFactors = T)
songs <- subset(songs, select = -cover_url)
artists <- read.csv("artists_cleaned.csv",header=T,stringsAsFactors = T)
is_ascii <- function(x) {
x_ascii <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = "")
nchar(x_ascii) == nchar(x) & !is.na(x_ascii)
}
ascii_indices <- is_ascii(artists$artist_mb)
artists_ascii <- artists[ascii_indices, ]
artists$artist_mb <- as.character(artists$artist_mb)
ascii_indices <- is_ascii(artists$artist_mb)
artists_ascii <- artists[ascii_indices, ]
n_total <- nrow(artists)
n_ascii <- nrow(artists_ascii)
artists_ascii$artist_mb <- trimws(artists_ascii$artist_mb)
artists_ascii$artist_mb <- tolower(artists_ascii$artist_mb)
ascii_artist_names <- unique(artists_ascii$artist_mb)
View(artists_ascii)
artists$tags_mb <- as.character(artists$tags_mb)
artists$tags_lastfm <- as.character(artists$tags_lastfm)
artists$tags_mb[is.na(artists$tags_mb)] <- ""
artists$tags_lastfm[is.na(artists$tags_lastfm)] <- ""
artists$tags_mb[artists$tags_mb == "Not Found"] <- ""
artists$tags_lastfm[artists$tags_lastfm == "Not Found"] <- ""
artists <- artists[
(artists$tags_mb != "") | (artists$tags_lastfm != ""),
]
artists$artist_mb <- tolower(trimws(artists$artist_mb))
View(artists)
artists <- subset(artists, select = -c(tags_mb))
write.csv(artists, "artists_cleaned.csv", row.names = FALSE)
artists <- read.csv("artists_cleaned.csv",header=T,stringsAsFactors = T)
artists <- artists[1:50000,]
write.csv(artists, "artists_cleaned.csv", row.names = FALSE)
artists <- artists[!duplicated(artists$artist_mb),]
write.csv(artists, "artists_cleaned.csv", row.names = FALSE)
songs <- read.csv("Spotify Most Streamed Songs.csv",header=T,stringsAsFactors = T)
songs_artists <- merge(songs, artists, by.x="artist", by.y="artist_mb", all.x=TRUE)
View(songs)
songs_artists <- merge(songs, artists, by.x="artist.s.name", by.y="artist_mb", all.x=TRUE)
songs_artists <- merge(songs, artists, by.x="s.name", by.y="artist_mb", all.x=TRUE)
songs_artists <- merge(songs, artists, by.x="name", by.y="artist_mb", all.x=TRUE)
songs_artists <- merge(songs, artists, by.x="artist.s_name", by.y="artist_mb", all.x=TRUE)
songs_artists <- merge(songs, artists, by.x="s_name", by.y="artist_mb", all.x=TRUE)
songs_artists <- merge(songs, artists, by.x="_name", by.y="artist_mb", all.x=TRUE)
songs[1,]
artists[1,]
songs$artist_clean <- tolower(gsub("\\s+", "", songs$artist.s._name))
artists$artist_clean <- tolower(gsub("\\s+", "", artists$artist_mb))
merged_dataset <- merge(songs, artists, by = "artist_clean", all.x = TRUE)
View(merged_dataset)
merged_dataset <- subset(merged_dataset, select = -c(artist_clean,cover_url)
merged_dataset <- subset(merged_dataset, select = -c(artist_clean,cover_url))
merged_dataset <- subset(merged_dataset, select = -c(artist_clean,cover_url))
write.csv(merged_dataset, "dataset.csv", row.names = FALSE)
dataset <- read.csv("dataset.csv",header=T,stringsAsFactors = T)
View(dataset)
