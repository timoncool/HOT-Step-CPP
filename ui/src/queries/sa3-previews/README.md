# SA3 checkpoint previews

`queries.ts` owns the SA3 preview list for one run (`GET /datasets/:id/sa3-previews?run=`).
Keys sit in the training-runs area as `training-runs/sa3-previews` and carry the
worker, the dataset id and the run, so a read goes where its cache entry says.
The transport is `services/sa3PreviewApi.ts`; audio plays from each record's
server-validated `audioUrl`, never a path built in the browser.

The list polls while any cell is `queued` or `rendering`, or while the caller
says the training job is active (new rows can appear). `renderSa3PreviewCommand`
runs once, never retries, and on success invalidates only that worker's
previews for that dataset (`sa3PreviewKeys.dataset`). Cadence, prompts, seconds
and seeds are form state in the SA3 card until a job starts; the server echoes
the run's own settings in the list response.

Manual renders and retries send only the run and step (`manualRenderRequest`), so the
server uses the selected run's own saved settings, never the card's form values.
