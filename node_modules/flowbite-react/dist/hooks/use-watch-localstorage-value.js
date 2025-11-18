'use client';
import { useEffect } from 'react';

function useWatchLocalStorageValue({
  key: watchKey,
  onChange
}) {
  function handleStorageChange({ key, newValue }) {
    if (key === watchKey) onChange(newValue);
  }
  useEffect(() => {
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);
}

export { useWatchLocalStorageValue };
//# sourceMappingURL=use-watch-localstorage-value.js.map
