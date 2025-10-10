<script>
async function apiGet(path){
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} failed`);
  return r.json();
}

async function apiPut(path, body){
  const r = await fetch(path, {
    method: "PUT",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return r.json();
}
</script>
