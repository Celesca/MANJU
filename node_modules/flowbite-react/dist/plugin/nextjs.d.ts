import type { NextConfig } from "next";
interface ConfigurationOptions {
    defaultConfig: NextConfig;
}
type ConfigurationCallback = (phase: string, options: ConfigurationOptions) => NextConfig | Promise<NextConfig>;
export default function withFlowbiteReact(nextConfig?: NextConfig | ConfigurationCallback): ConfigurationCallback;
export {};
