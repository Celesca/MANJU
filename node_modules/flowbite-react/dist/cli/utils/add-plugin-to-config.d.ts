/**
 * Add a plugin to the configuration file.
 * @param {Object} options - The options for adding the plugin.
 * @param {string} options.configKey - The key in the configuration file where the plugin will be added.
 * @param {string} options.configPath - The path to the configuration file.
 * @param {string} options.pluginImportPath - The import path of the plugin.
 * @param {string} options.pluginName - The name of the plugin.
 */
export declare function addPluginToConfig({ configKey, configPath, pluginImportPath, pluginName, }: {
    configKey: string;
    configPath: string;
    pluginImportPath: string;
    pluginName: string;
}): Promise<void>;
