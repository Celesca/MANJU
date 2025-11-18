'use strict';

async function main(argv) {
  const [command, subCommand] = argv.map((arg) => arg.trim());
  try {
    if (command === "build") {
      const { build } = await Promise.resolve().then(function () { return require('./commands/build.cjs'); });
      await build();
    }
    if (command === "create") {
      const { create } = await Promise.resolve().then(function () { return require('./commands/create.cjs'); });
      await create(subCommand);
    }
    if (command === "dev") {
      const { dev } = await Promise.resolve().then(function () { return require('./commands/dev.cjs'); });
      await dev();
    }
    if (command === "help" || command === "--help") {
      const { help } = await Promise.resolve().then(function () { return require('./commands/help.cjs'); });
      await help();
    }
    if (command === "init") {
      const { init } = await Promise.resolve().then(function () { return require('./commands/init.cjs'); });
      await init();
    }
    if (command === "install") {
      const { installPackage } = await Promise.resolve().then(function () { return require('./commands/install.cjs'); });
      await installPackage();
    }
    if (command === "migrate") {
      const { migrate } = await Promise.resolve().then(function () { return require('./commands/migrate.cjs'); });
      await migrate();
    }
    if (command === "register") {
      const { register } = await Promise.resolve().then(function () { return require('./commands/register.cjs'); });
      await register();
    }
    if (command === "setup") {
      if (subCommand === "plugin") {
        const { setupPlugin } = await Promise.resolve().then(function () { return require('./commands/setup-plugin.cjs'); });
        await setupPlugin();
      }
      if (subCommand === "tailwindcss") {
        const { setupTailwind } = await Promise.resolve().then(function () { return require('./commands/setup-tailwind.cjs'); });
        await setupTailwind();
      }
      if (subCommand === "vscode") {
        const { setupVSCode } = await Promise.resolve().then(function () { return require('./commands/setup-vscode.cjs'); });
        await setupVSCode();
      }
    }
    if (!["build", "create", "dev", "help", "--help", "init", "install", "migrate", "register", "setup"].includes(command)) {
      console.error(`Unknown command: ${command}`);
      const { help } = await Promise.resolve().then(function () { return require('./commands/help.cjs'); });
      await help();
      process.exit(1);
    }
    if (command === "setup" && !["plugin", "tailwindcss", "vscode"].includes(subCommand)) {
      console.error(`Unknown subcommand: ${subCommand}`);
      const { helpSetup } = await Promise.resolve().then(function () { return require('./commands/help.cjs'); });
      await helpSetup();
      process.exit(1);
    }
  } catch (error) {
    console.error(`Error executing command ${command}:`, error);
    process.exit(1);
  }
}

exports.main = main;
//# sourceMappingURL=main.cjs.map
