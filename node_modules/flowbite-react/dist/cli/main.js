async function main(argv) {
  const [command, subCommand] = argv.map((arg) => arg.trim());
  try {
    if (command === "build") {
      const { build } = await import('./commands/build.js');
      await build();
    }
    if (command === "create") {
      const { create } = await import('./commands/create.js');
      await create(subCommand);
    }
    if (command === "dev") {
      const { dev } = await import('./commands/dev.js');
      await dev();
    }
    if (command === "help" || command === "--help") {
      const { help } = await import('./commands/help.js');
      await help();
    }
    if (command === "init") {
      const { init } = await import('./commands/init.js');
      await init();
    }
    if (command === "install") {
      const { installPackage } = await import('./commands/install.js');
      await installPackage();
    }
    if (command === "migrate") {
      const { migrate } = await import('./commands/migrate.js');
      await migrate();
    }
    if (command === "register") {
      const { register } = await import('./commands/register.js');
      await register();
    }
    if (command === "setup") {
      if (subCommand === "plugin") {
        const { setupPlugin } = await import('./commands/setup-plugin.js');
        await setupPlugin();
      }
      if (subCommand === "tailwindcss") {
        const { setupTailwind } = await import('./commands/setup-tailwind.js');
        await setupTailwind();
      }
      if (subCommand === "vscode") {
        const { setupVSCode } = await import('./commands/setup-vscode.js');
        await setupVSCode();
      }
    }
    if (!["build", "create", "dev", "help", "--help", "init", "install", "migrate", "register", "setup"].includes(command)) {
      console.error(`Unknown command: ${command}`);
      const { help } = await import('./commands/help.js');
      await help();
      process.exit(1);
    }
    if (command === "setup" && !["plugin", "tailwindcss", "vscode"].includes(subCommand)) {
      console.error(`Unknown subcommand: ${subCommand}`);
      const { helpSetup } = await import('./commands/help.js');
      await helpSetup();
      process.exit(1);
    }
  } catch (error) {
    console.error(`Error executing command ${command}:`, error);
    process.exit(1);
  }
}

export { main };
//# sourceMappingURL=main.js.map
