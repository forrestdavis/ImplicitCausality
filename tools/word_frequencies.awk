BEGIN {
    FS=" " }

{ 
    for (i=1; i<=NF; i++){
        word = tolower($i)
        words[word]++
    }
}
END {
    for (w in words)
        printf("%d %s\n", words[w], w)
}

