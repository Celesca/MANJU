function createConfig(input = {}) {
  return {
    $schema: input.$schema ?? "https://unpkg.com/flowbite-react/schema.json",
    components: input.components ?? [],
    dark: input.dark ?? true,
    path: input.path ?? "src/components",
    // TODO: infer from project
    prefix: input.prefix ?? "",
    rsc: input.rsc ?? true,
    tsx: input.tsx ?? true,
    version: input.version ?? 4
  };
}

export { createConfig };
//# sourceMappingURL=create-config.js.map
