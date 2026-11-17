"""Offline inference from coordinates and optional interval-valued event tags."""

import math

import numpy as np


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def prior_dead_start(dead_runs, tag_start, restart, margin):
    """Start of the latest outside/gap run that overlaps the tag's pre-restart window."""
    candidates = [start for start, stop in dead_runs
                  if start < restart and stop > max(0, tag_start - margin)]
    return max(candidates) if candidates else None


def tag_is_contradicted(start, stop, known, outside, speed, slow_speed, distance, teams,
                        control_radius, confirm_frames):
    """True only for the narrow case where live, visible, contested play fills the window.

    A tag is a claim that a stoppage exists. The claim is rejected only when every coordinate
    signal says the ball never stopped: it is continuously visible, never out, never slow, and
    both teams are within control distance around it. Any missing/out/slow frame is enough to
    keep trusting the tag.
    """
    if stop - start < 2:
        return False
    visible = known[start:stop]
    if not visible.all() or outside[start:stop].any():
        return False
    frames = speed[start:stop]
    if not np.isfinite(frames).all() or (frames <= slow_speed).any():
        return False
    near = distance[start:stop] <= control_radius
    home = (near & (teams == 'H')).any(axis=1)
    away = (near & (teams == 'A')).any(axis=1)
    return int((home & away).sum()) >= confirm_frames


def infer(observations, metadata, config, tags=None):
    """Return labels and observation-only diagnostics; never read files or answers."""
    fps = metadata['fps']
    scale = np.array([metadata['pitch_length'], metadata['pitch_width']])
    ball = observations[['ball_x', 'ball_y']].to_numpy(dtype=float) * scale
    n = len(ball)
    if not n:
        return [], {'restarts': [], 'confirmed_passes': [], 'visible_stoppages': [],
                    'long_gaps': [], 'moving_reappearances': []}
    slots = list(metadata['slot_teams'])
    teams = np.array([metadata['slot_teams'][s] for s in slots])
    players = np.stack([observations[[s + '_x', s + '_y']].to_numpy(dtype=float)
                        for s in slots], axis=1) * scale
    known = np.isfinite(ball).all(axis=1)
    radius = config['ball_radius_m']
    outside = known & ((ball < -radius) | (ball > scale + radius)).any(axis=1)
    offset = ball[:, None] - players
    distance = np.linalg.norm(offset, axis=2)
    relative = np.linalg.norm(np.diff(offset, axis=0), axis=2) * fps
    speed = np.r_[np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps, np.nan]

    # Keep the controlling slot so two confirmed teammates can prove a restart pass.
    stable = (distance <= config['control_radius_m']) & ~outside[:, None]
    stable[:-1] &= relative <= config['relative_speed_m_s']
    stable[-1] = False
    confirm = math.ceil(config['confirm_s'] * fps)
    confirmed = np.zeros_like(stable)
    for player in range(len(slots)):
        for start, stop in runs(stable[:, player]):
            if stop - start >= confirm:
                confirmed[start:stop - confirm + 1, player] = True
    evidence, controls = {}, []
    for player in range(len(slots)):
        for start, stop in runs(stable[:, player]):
            if (stop - start >= confirm
                    and not confirmed[start, teams != teams[player]].any()):
                controls.append((int(start), int(stop), player))
                if start:
                    evidence.setdefault(int(start), set()).add(str(teams[player]))

    forced_dead = outside.copy()
    gaps = []
    for start, stop in runs(~known):
        if (stop - start) / fps > config['gap_s']:
            forced_dead[start:stop] = True
            gaps.append([int(start), int(stop)])

    slow = speed <= config['slow_speed_m_s']
    slow_start = np.full(n, -1, dtype=int)
    for start, stop in runs(slow):
        slow_start[start:stop] = start

    relevant_tags = [t for t in (tags or []) if t.get('code') == 'SET PIECES']
    trusted = bool(config.get('trusted_tags', False))
    min_slow = math.ceil(config.get('trusted_tag_min_slow_s', 0.5) * fps)
    dead_runs = [(int(start), int(stop)) for start, stop in runs(forced_dead)]
    tags_used, tags_overridden = set(), set()
    team_conflicts = 0

    margin = math.ceil(config['tag_margin_s'] * fps)
    rise = math.ceil(config['release_rise_s'] * fps)
    separate = math.ceil(config['separation_s'] * fps)
    travel = math.ceil(config['restart_flight_s'] * fps)
    refractory = math.ceil(config['release_merge_s'] * fps)
    restarts, visible_stoppages = {}, []
    trusted_candidates = []
    last_launch = -refractory - 1
    for launch in np.flatnonzero(speed >= config['release_speed_m_s']):
        launch = int(launch)
        if launch - last_launch <= refractory or launch + separate >= n:
            continue
        previous = speed[max(0, launch - rise):launch]
        finite = previous[np.isfinite(previous)]
        after_gap = launch > 0 and not known[launch - 1]
        from_slow = (len(finite) and finite.min() <= config['slow_speed_m_s']
                     and speed[launch] - finite.min() >= config['release_gain_m_s'])
        if not (after_gap or from_slow):
            continue
        if not known[launch:launch + separate + 1].all():
            continue
        # A restored observation can already be several samples into the release.
        reach = config['control_radius_m']
        if after_gap:
            reach += speed[launch] * config['gap_release_grace_s']
        # Attribute the kick before acceleration moves the ball away from its source.
        slow_before = np.flatnonzero(previous <= config['slow_speed_m_s'])
        actor_frame = (launch if after_gap else
                       max(0, launch - rise) + int(slow_before[-1]))
        candidates = np.flatnonzero(
            (distance[actor_frame] <= reach)
            & (distance[launch + separate] - distance[actor_frame]
               >= config['separation_m']))
        if not len(candidates):
            continue
        order = candidates[np.argsort(distance[actor_frame, candidates])]
        actor = order[0]
        opponents = order[teams[order] != teams[actor]]
        if len(opponents):
            lead = distance[actor_frame, opponents[0]] - distance[actor_frame, actor]
            if lead == 0 or (after_gap and lead < config['actor_margin_m']):
                continue
        team = str(teams[actor])
        restart = launch
        if outside[launch]:
            # Throw-ins become live on entry, not on the release outside the pitch.
            entry = launch
            while entry < min(n, launch + travel) and known[entry] and outside[entry]:
                entry += 1
            if entry >= min(n, launch + travel) or not known[entry]:
                continue
            restart = entry
        candidate_support = [t for t in relevant_tags
                             if t['start_frame'] - margin <= restart < t['end_frame'] + margin]
        support = candidate_support
        coordinate_team = team
        near_teams = {str(teams[player]) for player in np.flatnonzero(
            distance[actor_frame] <= config['control_radius_m'])}
        coordinate_clear = len(near_teams) == 1
        team_source = 'coordinates'
        if trusted:
            tag_supported = False
            tag_conflict = False
        else:
            matching = [t for t in support if t.get('team') in (None, coordinate_team)]
            tag_supported = bool(matching)
            tag_conflict = bool(support and not matching)
        # A tag cannot supply a release or override an observed release actor.
        record = {'frame': int(restart), 'launch': launch, 'team': team,
                  'slot': slots[actor], 'coordinate_team': coordinate_team,
                  'team_source': team_source, 'tag_supported': tag_supported,
                  'tag_conflict': tag_conflict, 'tag_used': False, 'tag_overridden': False,
                  'after_gap': bool(after_gap), 'actor_frame': actor_frame}
        restarts[restart] = record
        last_launch = launch
        anchor = launch
        # A long gap can hide the latter part of an already observed slow wait.
        # Use the observed pre-gap interval, never a velocity across missing samples.
        tag_evidence = support if trusted else matching
        if tag_evidence and launch > 0 and slow_start[launch - 1] < 0:
            preceding = [g for g in gaps if 0 <= launch - g[1] <= rise]
            if preceding:
                anchor = preceding[-1][0] - 1
        before_transition = anchor
        # Permit at most two observed transition intervals immediately before release.
        while (anchor > 0 and before_transition - anchor < 2
               and known[anchor - 1] and slow_start[anchor - 1] < 0):
            anchor -= 1
        if not trusted and matching and anchor > 0 and slow_start[anchor - 1] >= 0:
            start = int(slow_start[anchor - 1])
            if ((anchor - start) / fps >= config['tagged_slow_s']
                    and not outside[start:launch].any()):
                # Preserve an ordinary-hold allowance before assigning a visible pause.
                onset = start + math.ceil(config['visible_grace_s'] * fps)
                forced_dead[onset:restart] = True
                visible_stoppages.append({'start': onset, 'slow_start': start,
                                          'slow_stop': int(anchor), 'stop': int(restart),
                                          'evidence_before_gap': before_transition != launch,
                                          'transition_frames': before_transition - anchor, 'team': team})
        elif trusted and support:
            slow_frame = (int(slow_start[anchor - 1])
                          if anchor > 0 and slow_start[anchor - 1] >= 0 else None)
            record['slow_frame'] = slow_frame
            record['anchor'] = int(anchor)
            record['before_transition'] = int(before_transition)
            record['transition_frames'] = before_transition - anchor
            trusted_candidates.append(record)

    # Assign tags only after all candidates are known. A live pass inside the tag window must
    # not consume the tag before the terminal restart of the same dead-ball episode.
    if trusted:
        for tag in relevant_tags:
            tag_start = max(0, min(n - 1, int(tag['start_frame'])))
            tag_stop = min(n, int(tag['end_frame']))
            kickoff = 'kick off' in str(tag.get('type') or '').lower()
            eligible = []
            for record in trusted_candidates:
                frame = record['frame']
                if not (tag_start - margin <= frame < tag_stop + margin):
                    continue
                prior = prior_dead_start(dead_runs, tag_start, frame, margin)
                slow_frame = record.get('slow_frame')
                if prior is not None:
                    start, kind = int(prior), 'prior_dead'
                elif (slow_frame is not None and frame - slow_frame >= min_slow
                      and slow[tag_start]):
                    start, kind = int(slow_frame), 'slow'
                elif (slow_frame is not None and slow_frame >= tag_start
                      and frame - slow_frame >= min_slow):
                    start, kind = int(slow_frame), 'slow'
                elif kickoff:
                    start, kind = int(tag_start), 'kickoff'
                else:
                    continue
                eligible.append((start, kind, record))
            if not eligible:
                if tag_is_contradicted(tag_start, tag_stop, known, outside, speed,
                                       config['slow_speed_m_s'], distance, teams,
                                       config['control_radius_m'], confirm):
                    tags_overridden.add(id(tag))
                continue
            start, kind, chosen = eligible[0]
            coordinate_team = chosen['coordinate_team']
            near_teams = {str(teams[player]) for player in np.flatnonzero(
                distance[chosen['actor_frame']] <= config['control_radius_m'])}
            coordinate_clear = len(near_teams) == 1
            tag_team = str(tag['team']) if tag.get('team') else None
            team = coordinate_team
            team_source = 'coordinates'
            if tag_team and not coordinate_clear:
                team = tag_team
                team_source = 'tag'
            elif not coordinate_clear:
                team_source = 'coordinates_ambiguous'
            if tag_team and coordinate_clear and tag_team != coordinate_team:
                team_conflicts += 1
            chosen['team'] = team
            chosen['team_source'] = team_source
            chosen['tag_conflict'] = bool(tag_team and coordinate_clear
                                          and tag_team != coordinate_team)
            if start < chosen['frame']:
                forced_dead[start:chosen['frame']] = True
                tags_used.add(id(tag))
                chosen['tag_used'] = True
                chosen['tag_supported'] = True
                visible_stoppages.append({
                    'start': start, 'slow_start': chosen.get('slow_frame', -1),
                    'slow_stop': chosen['anchor'], 'stop': chosen['frame'],
                    'evidence_before_gap': chosen['before_transition'] != chosen['launch'],
                    'transition_frames': chosen['transition_frames'], 'team': team,
                    'team_source': team_source, 'tag_start': tag_start,
                    'tag_end': tag_stop, 'evidence': kind})

    moving_reappearances = []
    if config.get('end_gap_on_moving_ball', False):
        for _, stop in gaps:
            if stop <= 0 or stop >= n or stop in restarts:
                continue
            end = stop
            if outside[stop]:
                # A throw-in reappears outside the pitch; its restart is the entry frame.
                while (end < min(n, stop + travel) and known[end] and outside[end]):
                    end += 1
                if end >= min(n, stop + travel) or not known[end]:
                    continue
            first = min((c for c in controls if c[0] >= end),
                        key=lambda c: c[0], default=None)
            if first is None or first[0] - end > travel:
                continue
            first_speed = next((frame for frame in range(end, min(n, end + travel + 1))
                                if np.isfinite(speed[frame])), None)
            if first_speed is None or speed[first_speed] <= config['slow_speed_m_s']:
                continue
            # The moving ball must stay visible and in play until the first control; an
            # observed release, another gap or an exit keeps the original behavior.
            if (not known[end:first[0] + 1].all()
                    or outside[end:first[0] + 1].any()
                    or forced_dead[end:first[0] + 1].any()
                    or any(stop <= frame <= first[0] for frame in restarts)):
                continue
            tag_team = None
            if trusted:
                for tag in relevant_tags:
                    if not tag.get('team'):
                        continue
                    tag_start = max(0, min(n - 1, int(tag['start_frame'])))
                    tag_stop = min(n, int(tag['end_frame']))
                    if (tag_start - margin <= end < tag_stop + margin
                            or (tag_start <= first[0] and tag_stop >= end)):
                        tag_team = str(tag['team'])
                        break
            team = tag_team or str(teams[first[2]])
            restarts[end] = {
                'frame': int(end), 'launch': int(stop), 'team': team,
                'slot': None, 'coordinate_team': str(teams[first[2]]),
                'team_source': 'tag' if tag_team else 'first_control',
                'tag_supported': bool(tag_team), 'tag_used': False,
                'tag_overridden': False, 'after_gap': True, 'revived': True,
                'moving_reappearance': True, 'control_start': int(first[0])}
            moving_reappearances.append(restarts[end])

    passes = {}
    ordered = sorted(controls)
    for first, second in zip(ordered, ordered[1:]):
        start, launch, source = first
        arrival, _, receiver = second
        if (source == receiver or teams[source] != teams[receiver]
                or not launch < arrival <= launch + travel
                or not known[start:arrival + confirm].all()
                or not np.isfinite(distance[launch:arrival + confirm, source]).all()
                or distance[arrival, source] - distance[launch - 1, source] < config['separation_m']
                or forced_dead[launch:arrival + confirm].any()):
            continue
        passes[launch] = {'frame': launch, 'team': str(teams[source]),
                          'source_slot': slots[source], 'receiver_slot': slots[receiver],
                          'source_start': start, 'source_stop': launch,
                          'receiver_start': arrival}

    state, labels, dead_since = 'D', [], 0
    for frame in range(n):
        if forced_dead[frame]:
            if state != 'D':
                dead_since = frame
            state = 'D'
        elif (state == 'D' and frame in passes
              and frame - max(dead_since, passes[frame]['source_start']) >= confirm):
            state = passes[frame]['team']
        elif state == 'D' and frame in restarts:
            state = restarts[frame]['team']
        elif state != 'D' and known[frame] and len(evidence.get(frame, ())) == 1:
            state = next(iter(evidence[frame]))
        labels.append(state)
    return labels, {'restarts': list(restarts.values()),
                    'confirmed_passes': list(passes.values()),
                    'visible_stoppages': visible_stoppages, 'long_gaps': gaps,
                    'moving_reappearances': moving_reappearances,
                    'tags_seen': len(relevant_tags), 'tags_used': len(tags_used),
                    'tags_overridden': len(tags_overridden), 'tag_team_conflicts': team_conflicts,
                    'tags_unmatched': len(relevant_tags) - len(tags_used | tags_overridden)}


def predict(observations, metadata, config, tags=None):
    """Return one H/A/D per frame; absent tags select the strict coordinate branch."""
    return infer(observations, metadata, config, tags)[0]
