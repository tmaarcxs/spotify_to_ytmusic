#!/usr/bin/env python3

import json
import sys
import os
import time
import re

from ytmusicapi import YTMusic
from typing import Optional, Union, Iterator, Dict, List
from collections import namedtuple
from dataclasses import dataclass, field


SongInfo = namedtuple("SongInfo", ["title", "artist", "album"])


def get_ytmusic() -> YTMusic:
    """
    @@@
    """
    if not os.path.exists("oauth.json"):
        print("ERROR: No file 'oauth.json' exists in the current directory.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)

    try:
        return YTMusic("oauth.json")
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: JSON Decode error while trying start YTMusic: {e}")
        print("       This typically means a problem with a 'oauth.json' file.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)


def _ytmusic_create_playlist(
    yt: YTMusic, title: str, description: str, privacy_status: str = "PRIVATE"
) -> str:
    """Wrapper on ytmusic.create_playlist

    This wrapper does retries with back-off because sometimes YouTube Music will
    rate limit requests or otherwise fail.

    privacy_status can be: PRIVATE, PUBLIC, or UNLISTED
    """

    def _create(
        yt: YTMusic, title: str, description: str, privacy_status: str
    ) -> Union[str, dict]:
        exception_sleep = 5
        for _ in range(10):
            try:
                """Create a playlist on YTMusic, retrying if it fails."""
                id = yt.create_playlist(
                    title=title, description=description, privacy_status=privacy_status
                )
                return id
            except Exception as e:
                print(
                    f"ERROR: (Retrying create_playlist: {title}) {e} in {exception_sleep} seconds"
                )
                time.sleep(exception_sleep)
                exception_sleep *= 2

        return {
            "s2yt error": 'ERROR: Could not create playlist "{title}" after multiple retries'
        }

    id = _create(yt, title, description, privacy_status)
    #  create_playlist returns a dict if there was an error
    if isinstance(id, dict):
        print(f"ERROR: Failed to create playlist (name: {title}): {id}")
        sys.exit(1)

    time.sleep(1)  # seems to be needed to avoid missing playlist ID error

    return id


def load_playlists_json(filename: str = "playlists.json", encoding: str = "utf-8"):
    """Load the `playlists.json` Spotify playlist file"""
    return json.load(open(filename, "r", encoding=encoding))


def create_playlist(pl_name: str, privacy_status: str = "PRIVATE") -> None:
    """Create a YTMusic playlist


    Args:
        `pl_name` (str): The name of the playlist to create. It should be different to "".

        `privacy_status` (str: PRIVATE, PUBLIC, UNLISTED) The privacy setting of created playlist.
    """
    yt = get_ytmusic()

    id = _ytmusic_create_playlist(
        yt, title=pl_name, description=pl_name, privacy_status=privacy_status
    )
    print(f"Playlist ID: {id}")


def iter_spotify_liked_albums(
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
) -> Iterator[SongInfo]:
    """Songs from liked albums on Spotify."""
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    if "albums" not in spotify_pls:
        return None

    for album in [x["album"] for x in spotify_pls["albums"]]:
        for track in album["tracks"]["items"]:
            yield SongInfo(track["name"], track["artists"][0]["name"], album["name"])


def iter_spotify_playlist(
    src_pl_id: Optional[str] = None,
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
    reverse_playlist: bool = False,
) -> Iterator[SongInfo]:
    """Songs from a specific album ("Liked Songs" if None)

    Args:
        `src_pl_id` (Optional[str], optional): The ID of the source playlist. Defaults to None.
        `spotify_playlist_file` (str, optional): The path to the playlists backup files. Defaults to "playlists.json".
        `spotify_encoding` (str, optional): Characters encoding. Defaults to "utf-8".
        `reverse_playlist` (bool, optional): Is the playlist reversed when loading?  Defaults to False.

    Yields:
        Iterator[SongInfo]: The song's information
    """
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    def find_spotify_playlist(spotify_pls: Dict, src_pl_id: Union[str, None]) -> Dict:
        """Return the spotify playlist that matches the `src_pl_id`.

        Args:
            `spotify_pls`: The playlist datastrcuture saved by spotify-backup.
            `src_pl_id`: The ID of a playlist to find, or None for the "Liked Songs" playlist.
        """
        for src_pl in spotify_pls["playlists"]:
            if src_pl_id is None and str(src_pl.get("name")) == "Liked Songs":
                return src_pl
            if src_pl_id is not None and str(src_pl.get("id")) == src_pl_id:
                return src_pl
        raise ValueError(f"Could not find Spotify playlist {src_pl_id}")

    src_pl = find_spotify_playlist(spotify_pls, src_pl_id)
    src_pl_name = src_pl["name"]

    print(f"== Spotify Playlist: {src_pl_name}")

    # Sort tracks by added_at date before processing them
    pl_tracks = sorted(
        src_pl["tracks"], key=lambda x: x.get("added_at", ""), reverse=reverse_playlist
    )
    print(f"DEBUG: Sorted tracks by added_at date, reverse={reverse_playlist}")

    for src_track in pl_tracks:
        if src_track["track"] is None:
            print(
                f"WARNING: Spotify track seems to be malformed, Skipping.  Track: {src_track!r}"
            )
            continue

        try:
            src_album_name = src_track["track"]["album"]["name"]
            src_track_artist = src_track["track"]["artists"][0]["name"]
        except TypeError as e:
            print(f"ERROR: Spotify track seems to be malformed.  Track: {src_track!r}")
            raise e
        src_track_name = src_track["track"]["name"]

        # Print the date the track was added for debugging
        added_date = src_track.get("added_at", "Unknown date")
        print(f"DEBUG: Track {src_track_name} was added at {added_date}")

        yield SongInfo(src_track_name, src_track_artist, src_album_name)


def get_playlist_id_by_name(yt: YTMusic, title: str) -> Optional[str]:
    """Look up a YTMusic playlist ID by name.

    Args:
        `yt` (YTMusic): _description_
        `title` (str): _description_

    Returns:
        Optional[str]: The playlist ID or None if not found.
    """
    #  ytmusicapi seems to run into some situations where it gives a Traceback on listing playlists
    #  https://github.com/sigma67/ytmusicapi/issues/539
    try:
        playlists = yt.get_library_playlists(limit=5000)
    except KeyError as e:
        print("=" * 60)
        print(f"Attempting to look up playlist '{title}' failed with KeyError: {e}")
        print(
            "This is a bug in ytmusicapi that prevents 'copy_all_playlists' from working."
        )
        print(
            "You will need to manually copy playlists using s2yt_list_playlists and s2yt_copy_playlist"
        )
        print(
            "until this bug gets resolved.  Try `pip install --upgrade ytmusicapi` just to verify"
        )
        print("you have the latest version of that library.")
        print("=" * 60)
        raise

    for pl in playlists:
        if pl["title"] == title:
            return pl["playlistId"]

    return None


@dataclass
class ResearchDetails:
    query: Optional[str] = field(default=None)
    songs: Optional[List[Dict]] = field(default=None)
    suggestions: Optional[List[str]] = field(default=None)


def lookup_song(
    yt: YTMusic,
    track_name: str,
    artist_name: str,
    album_name,
    yt_search_algo: int,
    details: Optional[ResearchDetails] = None,
) -> dict:
    """Look up a song on YTMusic

    Given the Spotify track information, it does a lookup for songs
    by the track name and artist on YTMusic. Only returns proper YouTube Music songs,
    not YouTube videos.

    Args:
        `yt` (YTMusic)
        `track_name` (str): The name of the researched track
        `artist_name` (str): The name of the researched track's artist
        `album_name` (str): The name of the researched track's album
        `yt_search_algo` (int): 0 for exact matching, 1 for extended matching (search past 1st result), 2 for approximate matching (search more songs)
        `details` (ResearchDetails): If specified, more information about the search and the response will be populated for use by the caller.

    Raises:
        ValueError: If no track is found, it returns an error

    Returns:
        dict: The infos of the researched song
    """

    # Helper function to check if a result is a valid song (not a YouTube video)
    def is_valid_song(song_result):
        # A true YouTube Music song MUST have all these attributes
        has_album = "album" in song_result and song_result["album"] is not None

        # Check if it has artist info structured like song results
        has_artists = (
            "artists" in song_result
            and isinstance(song_result["artists"], list)
            and len(song_result["artists"]) > 0
        )

        # A song must have both album and artists information
        return has_album and has_artists

    # Search directly for songs with the track and artist name
    query = f"{track_name} by {artist_name}"

    if details:
        details.query = query
        details.suggestions = yt.get_search_suggestions(query=query)

    # Always filter by songs
    songs = yt.search(query=query, filter="songs")

    # Filter out results that don't match our criteria for being a song
    valid_songs = []
    for song in songs:
        if is_valid_song(song):
            valid_songs.append(song)

    if details:
        details.songs = valid_songs

    # If no valid songs were found, try a different query
    if not valid_songs:
        # Try another query formation
        alt_query = f"{artist_name} {track_name}"
        alt_songs = yt.search(query=alt_query, filter="songs")

        for song in alt_songs:
            if is_valid_song(song):
                valid_songs.append(song)

        # If still no valid songs, try with album name in query
        if not valid_songs and album_name:
            album_query = f"{track_name} {artist_name} {album_name}"
            album_songs = yt.search(query=album_query, filter="songs")

            for song in album_songs:
                if is_valid_song(song):
                    valid_songs.append(song)

        if not valid_songs:
            raise ValueError(
                f"Did not find any valid YouTube Music songs for {track_name} by {artist_name} from {album_name}"
            )

    match yt_search_algo:
        case 0:
            # Return the first valid song
            return valid_songs[0]

        case 1:
            # Try to find an exact match among valid songs
            for song in valid_songs:
                if (
                    song["title"] == track_name
                    and song["artists"][0]["name"] == artist_name
                ):
                    return song

            # If no exact match, return the first valid song
            if valid_songs:
                return valid_songs[0]

            raise ValueError(
                f"Did not find {track_name} by {artist_name} from {album_name}"
            )

        case 2:
            # Try fuzzy matching among valid songs
            for song in valid_songs:
                # Remove everything in brackets in the song title
                song_title_without_brackets = re.sub(
                    r"[\[(].*?[])]", "", song["title"]
                ).strip()
                if (
                    song_title_without_brackets == track_name
                    or song_title_without_brackets in track_name
                    or track_name in song_title_without_brackets
                ) and (
                    song["artists"][0]["name"] == artist_name
                    or artist_name in song["artists"][0]["name"]
                ):
                    return song

            # Try an extended search for songs with a broader query
            extended_query = f"{track_name} {artist_name}"
            extended_songs = yt.search(query=extended_query, filter="songs", limit=20)

            valid_extended_songs = []
            for song in extended_songs:
                if is_valid_song(song):
                    valid_extended_songs.append(song)

            # Look for approximate matches in extended search
            for song in valid_extended_songs:
                song_title = song["title"].lower()
                track_name_lower = track_name.lower()
                artist_name_lower = artist_name.lower()

                if track_name_lower in song_title and (
                    artist_name_lower in song_title
                    or any(
                        artist_name_lower in artist["name"].lower()
                        for artist in song["artists"]
                    )
                ):
                    return song

            # If we still can't find anything, use the first valid song from the original search
            if valid_songs:
                return valid_songs[0]

            # Last resort - we have no matches
            raise ValueError(
                f"Did not find {track_name} by {artist_name} from {album_name}"
            )


def copier(
    src_tracks: Iterator[SongInfo],
    dst_pl_id: Optional[str] = None,
    dry_run: bool = False,
    track_sleep: float = 0.1,
    yt_search_algo: int = 0,
    *,
    yt: Optional[YTMusic] = None,
    reverse_order: bool = False,
):
    """
    Copy songs from src_tracks to dst_pl_id YTMusic playlist

    Args:
        src_tracks: Iterator of SongInfo objects to copy
        dst_pl_id: YouTube Music playlist ID to copy to
        dry_run: If True, don't actually add songs
        track_sleep: Time to sleep between tracks
        yt_search_algo: Algorithm to use for searching (0=exact, 1=extended, 2=approximate)
        yt: YTMusic instance (will create one if None)
        reverse_order: If True, reverse the order of tracks before adding to YouTube Music
    """
    if yt is None:
        yt = get_ytmusic()

    print(
        f"DEBUG: Copier started with parameters - dst_pl_id: {dst_pl_id}, dry_run: {dry_run}, reverse_order: {reverse_order}"
    )

    if dst_pl_id is not None:
        try:
            yt_pl = yt.get_playlist(playlistId=dst_pl_id)
            print(f"DEBUG: Successfully retrieved playlist with ID {dst_pl_id}")
        except Exception as e:
            print(f"ERROR: Unable to find YTMusic playlist {dst_pl_id}: {e}")
            print(
                "       Make sure the YTMusic playlist ID is correct, it should be something like "
            )
            print("      'PL_DhcdsaJ7echjfdsaJFhdsWUd73HJFca'")
            sys.exit(1)
        print(f"== Youtube Playlist: {yt_pl['title']}")

    # Track statistics
    songs_added = 0
    duplicate_count = 0
    error_count = 0
    tracks_added_set = set()

    # If we need to reverse the order, we need to collect all tracks first
    tracks_to_process = []

    print("Searching for tracks...")
    track_count = 0
    for src_track in src_tracks:
        track_count += 1
        print(f"Spotify:   {src_track.title} - {src_track.artist} - {src_track.album}")

        try:
            print(
                f"DEBUG: Looking up song {src_track.title} by {src_track.artist} with algorithm {yt_search_algo}"
            )
            dst_track = lookup_song(
                yt, src_track.title, src_track.artist, src_track.album, yt_search_algo
            )
        except Exception as e:
            print(f"ERROR: Unable to look up song on YTMusic: {e}")
            error_count += 1
            continue

        yt_artist_name = "<Unknown>"
        if "artists" in dst_track and len(dst_track["artists"]) > 0:
            yt_artist_name = dst_track["artists"][0]["name"]
        print(
            f"  Youtube: {dst_track['title']} - {yt_artist_name} - {dst_track['album'] if 'album' in dst_track else '<Unknown>'}"
        )

        if dst_track["videoId"] in tracks_added_set:
            print("(DUPLICATE, this track has already been added)")
            duplicate_count += 1
            continue

        tracks_added_set.add(dst_track["videoId"])

        # If we need to reverse the order, we collect tracks first
        if reverse_order:
            print(f"DEBUG: Collecting track for later (reverse order is enabled)")
            tracks_to_process.append(dst_track)
        else:
            # Otherwise add each song immediately
            if not dry_run:
                print(f"DEBUG: Adding song immediately (reverse order is disabled)")
                success = add_single_song_to_playlist(yt, dst_pl_id, dst_track)
                if success:
                    songs_added += 1
                else:
                    error_count += 1
                if track_sleep:
                    time.sleep(track_sleep)
            else:
                print("DEBUG: Dry run enabled, not actually adding song")

    print(
        f"DEBUG: Found {track_count} tracks from Spotify, collected {len(tracks_to_process)} for reverse processing"
    )

    # If we collected tracks to reverse them, now process them
    if reverse_order and tracks_to_process:
        print(
            f"DEBUG: Reversing {len(tracks_to_process)} tracks order for YouTube Music..."
        )
        tracks_to_process.reverse()

        if not dry_run:
            print(f"Adding {len(tracks_to_process)} tracks to YouTube Music...")
            for dst_track in tracks_to_process:
                success = add_single_song_to_playlist(yt, dst_pl_id, dst_track)
                if success:
                    songs_added += 1
                else:
                    error_count += 1
                if track_sleep:
                    time.sleep(track_sleep)
        else:
            print(
                "DEBUG: Dry run enabled, not actually adding songs from reverse order collection"
            )

    print()
    print(
        f"Added {songs_added} tracks, encountered {duplicate_count} duplicates, {error_count} errors"
    )


def add_single_song_to_playlist(
    yt: YTMusic, playlist_id: Optional[str], track: dict
) -> bool:
    """
    Add a single song to a YouTube Music playlist with retry logic

    Args:
        yt: YTMusic instance
        playlist_id: ID of the playlist to add to, or None for liked songs
        track: Track information from YTMusic API

    Returns:
        bool: Whether the song was successfully added
    """
    print(
        f"DEBUG: Attempting to add song {track['title']} (ID: {track['videoId']}) to playlist {playlist_id}"
    )
    exception_sleep = 5
    for attempt in range(10):
        try:
            if playlist_id is not None:
                # Add to specified playlist
                print(
                    f"DEBUG: API call - Adding video ID {track['videoId']} to playlist {playlist_id}"
                )
                # Setting source_playlist=None makes YouTube Music add the song to the end of the playlist
                result = yt.add_playlist_items(
                    playlistId=playlist_id,
                    videoIds=[track["videoId"]],
                    duplicates=False,
                    source_playlist=None,
                )
                print(f"DEBUG: API response: {result}")
                print(f"Added: {track['title']}")
            else:
                # For liked songs
                print(f"DEBUG: API call - Liking video ID {track['videoId']}")
                result = yt.rate_song(track["videoId"], "LIKE")
                print(f"DEBUG: API response: {result}")
                print(f"Liked: {track['title']}")
            return True
        except Exception as e:
            print(
                f"ERROR: (Retrying adding {track['title']}) {e} in {exception_sleep} seconds"
            )
            print(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
            time.sleep(exception_sleep)
            exception_sleep *= 2

    print(f"Failed to add {track['title']} after multiple retries")
    return False


def copy_playlist(
    spotify_playlist_id: str,
    ytmusic_playlist_id: str,
    spotify_playlists_encoding: str = "utf-8",
    dry_run: bool = False,
    track_sleep: float = 0.1,
    yt_search_algo: int = 0,
    reverse_playlist: bool = False,
    privacy_status: str = "PRIVATE",
):
    """
    Copy a Spotify playlist to a YTMusic playlist

    Args:
        spotify_playlist_id: Spotify playlist ID to copy from
        ytmusic_playlist_id: YouTube Music playlist ID to copy to (or +name to look up)
        spotify_playlists_encoding: Encoding of the playlists.json file
        dry_run: If True, don't actually add songs
        track_sleep: Time to sleep between tracks
        yt_search_algo: Algorithm to use for searching (0=exact, 1=extended, 2=approximate)
        reverse_playlist: If True, reverse the order of tracks from Spotify
        privacy_status: Privacy status of created playlist (PRIVATE, PUBLIC, UNLISTED)
    """
    print(
        f"DEBUG: copy_playlist started - spotify_id: {spotify_playlist_id}, yt_id: {ytmusic_playlist_id}"
    )
    print(
        f"DEBUG: Parameters - dry_run: {dry_run}, reverse_playlist: {reverse_playlist}"
    )
    print("Using search algo n°: ", yt_search_algo)
    yt = get_ytmusic()
    pl_name: str = ""

    if ytmusic_playlist_id and ytmusic_playlist_id.startswith("+"):
        pl_name = ytmusic_playlist_id[1:]

        ytmusic_playlist_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={ytmusic_playlist_id}")

    if ytmusic_playlist_id is None:
        if pl_name == "":
            print("No playlist name or ID provided, creating playlist...")
            spotify_pls: dict = load_playlists_json()
            for pl in spotify_pls["playlists"]:
                if len(pl.keys()) > 3 and pl["id"] == spotify_playlist_id:
                    pl_name = pl["name"]

        ytmusic_playlist_id = _ytmusic_create_playlist(
            yt,
            title=pl_name,
            description=pl_name,
            privacy_status=privacy_status,
        )

        #  create_playlist returns a dict if there was an error
        if isinstance(ytmusic_playlist_id, dict):
            print(f"ERROR: Failed to create playlist: {ytmusic_playlist_id}")
            sys.exit(1)
        print(f"NOTE: Created playlist '{pl_name}' with ID: {ytmusic_playlist_id}")

    # Since we're adding songs in batches now, we need to use the same behavior
    # for both Spotify and YouTube Music. The reverse_playlist parameter handles the
    # Spotify side, and we'll now pass a parameter to control the YouTube Music side.
    print(
        f"DEBUG: Calling copier with reverse_playlist={reverse_playlist}, reverse_order={reverse_playlist}"
    )
    copier(
        iter_spotify_playlist(
            spotify_playlist_id,
            spotify_encoding=spotify_playlists_encoding,
            reverse_playlist=reverse_playlist,
        ),
        ytmusic_playlist_id,
        dry_run,
        track_sleep,
        yt_search_algo,
        yt=yt,
        reverse_order=reverse_playlist,  # Invert the order for YouTube Music if needed
    )


def copy_all_playlists(
    track_sleep: float = 0.1,
    dry_run: bool = False,
    spotify_playlists_encoding: str = "utf-8",
    yt_search_algo: int = 0,
    reverse_playlist: bool = False,
    privacy_status: str = "PRIVATE",
):
    """
    Copy all Spotify playlists (except Liked Songs) to YTMusic playlists
    """
    spotify_pls = load_playlists_json()
    yt = get_ytmusic()

    for src_pl in spotify_pls["playlists"]:
        if str(src_pl.get("name")) == "Liked Songs":
            continue

        pl_name = src_pl["name"]
        if pl_name == "":
            pl_name = f"Unnamed Spotify Playlist {src_pl['id']}"

        dst_pl_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={dst_pl_id}")
        if dst_pl_id is None:
            dst_pl_id = _ytmusic_create_playlist(
                yt, title=pl_name, description=pl_name, privacy_status=privacy_status
            )

            #  create_playlist returns a dict if there was an error
            if isinstance(dst_pl_id, dict):
                print(f"ERROR: Failed to create playlist: {dst_pl_id}")
                sys.exit(1)
            print(f"NOTE: Created playlist '{pl_name}' with ID: {dst_pl_id}")

        copier(
            iter_spotify_playlist(
                src_pl["id"],
                spotify_encoding=spotify_playlists_encoding,
                reverse_playlist=reverse_playlist,
            ),
            dst_pl_id,
            dry_run,
            track_sleep,
            yt_search_algo,
            yt=yt,
            reverse_order=reverse_playlist,  # Invert the order for YouTube Music if needed
        )
        print("\nPlaylist done!\n")

    print("All done!")
