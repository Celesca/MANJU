const store = {
  dark: void 0,
  mode: void 0,
  prefix: void 0,
  version: void 0
};
function setStore(data) {
  if ("dark" in data) {
    store.dark = data.dark;
  }
  if ("mode" in data) {
    if (["light", "dark", "auto"].includes(data.mode)) {
      store.mode = data.mode;
    } else {
      console.warn(`Invalid mode value: ${data.mode}.
Available values: light, dark, auto`);
    }
  }
  if ("prefix" in data) {
    store.prefix = data.prefix;
  }
  if ("version" in data) {
    if (data.version === 3 || data.version === 4) {
      store.version = data.version;
    } else {
      console.warn(`Invalid version value: ${data.version}.
Available values: 3, 4`);
    }
  }
}
function getDark() {
  return store.dark;
}
function getMode() {
  return store.mode;
}
function getPrefix() {
  return store.prefix;
}
function getVersion() {
  return store.version;
}

export { getDark, getMode, getPrefix, getVersion, setStore };
//# sourceMappingURL=index.js.map
