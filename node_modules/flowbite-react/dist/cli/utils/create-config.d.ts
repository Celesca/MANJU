export interface Config {
    $schema: string;
    components: string[];
    dark: boolean;
    path: string;
    prefix: string;
    rsc: boolean;
    tsx: boolean;
    version: 3 | 4;
}
export declare function createConfig(input?: Partial<Config>): Config;
